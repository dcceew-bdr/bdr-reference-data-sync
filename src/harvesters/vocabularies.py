from itertools import chain
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Union, Awaitable
import asyncio
import rdflib
from rdflib.plugins.stores import sparqlstore
from rdflib.namespace import RDF, RDFS, DCAT, SKOS, OWL, DCTERMS
from rdflib.term import Identifier
from copy import deepcopy, copy
import httpx

from .local_fetch import get_broadest_concepts, get_all_concepts, get_concept_scheme_members, get_collection_members
from .sparql_fetch import sparql_describe, sparql_subjects, sparql_objects, sparql_concept_scheme_members, \
    sparql_collection_members, sparql_all_concepts
from src.harvesters.sparql_fetch import sparql_broadest_concepts
from ..voc_graph import VocabGraphDetails, TERN


async def async_task_association(task: Union[asyncio.Task, Awaitable], assoc: Any) -> Union[Exception, Tuple]:
    try:
        res = await task
    except Exception as e:
        return e  # Return exceptions, not a tuple.
    return res, assoc


class VocabHarvester:
    is_init: bool
    source_graph: rdflib.Graph
    vocab_type: rdflib.URIRef
    root_node: rdflib.URIRef
    root_node_details: rdflib.Graph # CBD of root_node
    concept_schemes: Set[rdflib.URIRef]
    concept_collections: Set[rdflib.URIRef]
    concepts: Set[rdflib.URIRef]
    broadest_concepts: Set[rdflib.URIRef]
    exclude_concepts: List[rdflib.URIRef]
    exclude_concept_schemes: List[rdflib.URIRef]
    exclude_concept_collections: List[rdflib.URIRef]
    include_concepts: List[rdflib.URIRef]
    include_concept_schemes: List[rdflib.URIRef]
    include_concept_collections: List[rdflib.URIRef]
    concept_scheme_concepts: Dict[rdflib.URIRef, Set[rdflib.URIRef]]
    collection_members: Dict[rdflib.URIRef, Set[rdflib.URIRef]]
    concept_maps: Dict[rdflib.URIRef, Dict]
    collection_maps: Dict[rdflib.URIRef, Dict]

    def __init__(self, source_graph: rdflib.Graph):
        self.is_init = False
        self.source_graph = source_graph
        self.concepts = set()
        self.broadest_concepts = set()
        self.concept_schemes = set()
        self.concept_collections = set()
        self.collection_members = {}
        self.concept_scheme_concepts = {}
        self.concept_maps = {}
        self.collection_maps = {}  # for mapping collections to parent collections
        self.reset_filters()

    def reset_filters(self):
        self.filtered_concepts = copy(self.concepts)
        self.filtered_concept_schemes = copy(self.concept_schemes)
        self.filtered_collections = copy(self.concept_collections)
        self.filtered_concept_scheme_concepts = deepcopy(self.concept_scheme_concepts)
        self.filtered_collection_members = deepcopy(self.collection_members)
        self.filtered_collection_maps = deepcopy(self.collection_maps)
        self.filtered_concept_maps = deepcopy(self.concept_maps)

    async def async_init(self):
        await self.identify_individuals()
        await self.identify_members()
        self.is_init = True

    @classmethod
    def build_from_source(cls, source: str) -> 'VocabHarvester':
        source_lower = source.lower()
        if source_lower.startswith("sparql:"):
            store = sparqlstore.SPARQLStore(query_endpoint=source[7:], method='POST', returnFormat='json')
            source_graph = rdflib.Graph(store=store, bind_namespaces="core")  # bind only core, so SPARQL prefixes in the sparql queries work properly
            vocab_harvester = SPARQLVocabHarvester(source_graph)
        elif source_lower.startswith("https:") or source_lower.startswith("http:"):
            source_graph = rdflib.Graph(bind_namespaces="core")  # bind only core, so SPARQL prefixes in the sparql queries work properly
            with httpx.Client() as client:
                try:
                    resp = client.get(source, headers={'Accept': 'text/turtle'}, follow_redirects=True)
                    resp.raise_for_status()
                except Exception as e:
                    print(e)
                    raise
                source_graph.parse(resp.read())
            vocab_harvester = LocalVocabHarvester(source_graph)
        elif source_lower.startswith("file://"):
            source_graph = rdflib.Graph(bind_namespaces="core")
            local_path = Path(source[7:])
            if not local_path.exists():
                raise RuntimeError(f"File {source} does not exist.")
            with open(local_path, "rb") as f:
                source_graph.parse(f)
            vocab_harvester = LocalVocabHarvester(source_graph)
        else:
            raise NotImplementedError(f"Unsupported vocab source type: {source}")
        return vocab_harvester

    def load_def(self, vocab_def: Dict[str, Any]):
        try:
            root_node = vocab_def["root_node"]
        except LookupError:
            raise RuntimeError("No root node identified for the SPARQL endpoint.")
        if not root_node:
            root_node = None
        else:
            root_node = rdflib.URIRef(root_node)
        self.root_node = root_node
        exclude_concept_schemes = vocab_def.get("exclude_concept_schemes", [])
        exclude_collections = vocab_def.get("exclude_collections", [])
        exclude_concepts = vocab_def.get("exclude_concepts", [])
        include_collections = vocab_def.get("include_collections", [])
        include_concept_schemes = vocab_def.get("include_concept_schemes", [])
        include_concepts = vocab_def.get("include_concepts", [])
        self.exclude_concept_schemes = [rdflib.URIRef(e) for e in exclude_concept_schemes]
        self.exclude_concept_collections = [rdflib.URIRef(e) for e in exclude_collections]
        self.exclude_concepts = [rdflib.URIRef(e) for e in exclude_concepts]
        self.include_concept_collections = [rdflib.URIRef(e) for e in include_collections]
        self.include_concept_schemes = [rdflib.URIRef(e) for e in include_concept_schemes]
        self.include_concepts = [rdflib.URIRef(e) for e in include_concepts]

    async def run_procedures(self) -> List[VocabGraphDetails]:
        """
        returns an individual Graph for each ConceptScheme
        load_def should be called immediately before calling this method
        """
        await self.determine_type()
        if not self.is_init:
            # Loads individuals and group memberships that apply for this whole source
            await self.async_init()
        self.reset_filters()
        self.filter()
        new_graphs = await self.harvest()
        return new_graphs

    async def determine_type(self):
        if self.root_node:
            try:
                self.root_node_details = await self.cbd(self.root_node)
            except Exception as e:
                print(e)
                raise RuntimeError("Could not find root node in that source vocab graph.")
            if len(self.root_node_details) == 0:
                raise RuntimeError("Could not find root node in that source vocab graph.")
            self.vocab_type = check_type(self.root_node_details, self.root_node)
        else:
            self.root_node_details = None
            self.vocab_type = None

    async def identify_individuals(self):
        self.concept_schemes = set(await self.subjects(RDF.type, SKOS.ConceptScheme))
        for concept_scheme in self.concept_schemes:
            self.concept_scheme_concepts[concept_scheme] = set()
        self.concept_collections = set(await self.subjects(RDF.type, SKOS.Collection))
        for concept_collection in self.concept_collections:
            self.collection_members[concept_collection] = set()
            self.collection_maps[concept_collection] = {'ancestors': set(), "defined_by": set()}
        self.concepts = set(await self.get_all_concepts())
        for concept in self.concepts:
            self.concept_maps[concept] = {'collections': set(), 'concept_schemes': set(), "defined_by": set()}
        self.broadest_concepts = set(await self.get_broadest_concepts())

    async def identify_members(self):
        loop = asyncio.get_event_loop()
        scheme_members_jobs = [loop.create_task(async_task_association(self.get_concept_scheme_members(s), s)) for s in
                               self.concept_schemes if isinstance(s, rdflib.URIRef)]

        await asyncio.sleep(0)  # yield loop to kick-start the jobs
        for i in await asyncio.gather(*scheme_members_jobs, return_exceptions=True):
            if isinstance(i, Exception):
                print(i)
                continue
            (t, n), s = i
            if s not in self.concept_scheme_concepts:
                print(f"Skipping unknown concept scheme: {s}")
                continue
            self.concept_scheme_concepts[s].update(t)
            self.concept_scheme_concepts[s].update(n)
            for top_c in t:
                if top_c not in self.concept_maps:
                    print(f"Skipping unknown top-concept from this concept scheme: {top_c}")
                    continue
                self.concept_maps[top_c]["concept_schemes"].add(s)
            for narrow_c in n:
                if narrow_c not in self.concept_maps:
                    print(f"Skipping unknown narrower-concept from this concept scheme: {narrow_c}")
                    continue
                self.concept_maps[narrow_c]["concept_schemes"].add(s)

        collection_members_jobs = [loop.create_task(async_task_association(self.get_collection_members(c), c)) for c in
                                   self.concept_collections if isinstance(c, rdflib.URIRef)]
        await asyncio.sleep(0)  # yield loop to kick-start the jobs
        for i in await asyncio.gather(*collection_members_jobs, return_exceptions=True):
            if isinstance(i, Exception):
                print(i)
                continue
            m, c = i
            if c not in self.collection_members:
                print(f"Skipping unknown concept collection: {c}")
                continue
            self.collection_members[c].update(m)
            for memb in m:
                if memb in self.collection_maps:
                    self.collection_maps[memb]["ancestors"].add(c)
                elif memb not in self.concept_maps:
                    print(f"Skipping unknown member from this concept collection: {memb}")
                    continue
                else:
                    self.concept_maps[memb]["collections"].add(c)

    def _filter_exclude_collections(self) -> Set[rdflib.URIRef]:
        extra_exclude_collections = set()
        more_extras = set()
        for _ in range(3):  # loop 3 times, to get 4-levels-deep collections-in-collections
            for e in chain(self.exclude_concept_collections, extra_exclude_collections):
                # First check for nested collections in the members of collections to be removed
                if e not in self.collection_members:
                    continue
                # already filtered out?
                if e not in self.filtered_collections:
                    continue
                members_to_check = self.collection_members[e]
                for member in members_to_check:
                    if member in self.collection_maps:
                        # member of a collection to remove is another collection, process these first
                        parents = self.collection_maps[member]["parentplus"]
                        if all((p in self.exclude_concept_collections or p in extra_exclude_collections or p in more_extras)
                               for p in parents):
                            more_extras.add(member)
                        else:
                            print("Found a collection member of a collection that cannot be excluded?")
                            print(member)
            extra_exclude_collections.update(more_extras)
            more_extras.clear()
        concepts_to_exclude = set()
        for e in chain(self.exclude_concept_collections, extra_exclude_collections):
            # Now check for concepts that are in these collections
            if e not in self.collection_members:
                continue
            # already filtered out?
            if (e not in self.filtered_collections) or (e not in self.filtered_collection_members):
                continue
            members_to_check = self.collection_members[e]
            for member in members_to_check:
                if member in self.concept_maps:
                    all_memberships = self.concept_maps[member]["collections"]
                    # member of a collection to remove is a concept
                    if all((a in self.exclude_concept_collections or a in extra_exclude_collections) for a in all_memberships):
                        concepts_to_exclude.add(member)
                    else:
                        print("Found a concept member of a collection that cannot be excluded?")
                        print(member)
            del self.filtered_collection_members[e]
            del self.filtered_collection_maps[e]
        remove_collections_set = set(self.exclude_concept_collections).union(extra_exclude_collections)
        self.filtered_collections.difference_update(remove_collections_set)
        return concepts_to_exclude

    def _filter_exclude_schemes(self) -> Set[rdflib.URIRef]:
        concepts_to_exclude = set()
        for e in self.exclude_concept_schemes:
            # Now check for concepts that are in excluded concept schemes
            if e not in self.concept_scheme_concepts:
                continue
            # already filtered out?
            if (e not in self.filtered_concept_schemes) or (e not in self.filtered_concept_scheme_concepts):
                continue
            members_to_check = self.concept_scheme_concepts[e]
            for member in members_to_check:
                if member in self.concept_maps:
                    schemes = self.concept_maps[member]["concept_schemes"]
                    # member of a collection to remove is a concept
                    if all(p in self.exclude_concept_schemes for p in schemes):
                        concepts_to_exclude.add(member)
                    else:
                        print("Found a concept member of a collection that cannot be excluded?")
                        print(member)
            del self.filtered_concept_scheme_concepts[e]
        self.filtered_concept_schemes.difference_update(self.exclude_concept_schemes)
        return concepts_to_exclude

    def _filter_include_collections(self) -> Set[rdflib.URIRef]:
        extra_include_collections = set()
        more_extras = set()
        for _ in range(3):  # loop 3 times, to get 4-levels-deep collections-in-collections
            for e in chain(self.include_concept_collections, extra_include_collections):
                # First check for nested collections in the members of collections to be whitelisted
                if e not in self.collection_members:
                    continue
                # already filtered out?
                if (e not in self.filtered_collections) or (e not in self.filtered_collection_members):
                    continue
                members_to_check = self.collection_members[e]
                for member in members_to_check:
                    if member in self.collection_maps:
                        # member of a collection to whitelist is another collection, do these first
                        parents = self.collection_maps[member]["ancestors"]
                        if any((p in self.include_concept_collections or p in extra_include_collections or p in more_extras)
                               for p in parents):
                            more_extras.add(member)
            extra_include_collections.update(more_extras)
            more_extras.clear()
        concepts_to_include = set()
        for e in chain(self.include_concept_collections, extra_include_collections):
            # Now check for concepts that are in these collections
            if e not in self.collection_members:
                continue
            # already filtered out?
            if (e not in self.filtered_collections) or (e not in self.filtered_collection_members):
                continue
            members_to_check = self.collection_members[e]
            for member in members_to_check:
                if member in self.concept_maps:
                    concepts_to_include.add(member)
        all_includes_collections = set(self.include_concept_collections).union(extra_include_collections)
        collections_to_remove = set(self.collection_members.keys()).difference(all_includes_collections)
        for c in collections_to_remove:
            if c in self.filtered_collections:  # This could have already been removed, in an exclude_collection
                del self.filtered_collection_maps[c]
                del self.filtered_collection_members[c]
        self.filtered_collections.intersection_update(all_includes_collections)
        return concepts_to_include

    def _filter_include_schemes(self) -> Set[rdflib.URIRef]:
        concepts_to_include = set()
        for e in self.include_concept_schemes:
            # Check for concepts that are in whitelisted concept schemes
            if e not in self.concept_scheme_concepts:
                continue
            members_to_check = self.concept_scheme_concepts[e]
            for member in members_to_check:
                if member in self.concept_maps:
                    concepts_to_include.add(member)

        schemes_to_remove = set(self.concept_scheme_concepts.keys()).difference(self.include_concept_schemes)
        for s in schemes_to_remove:
            del self.filtered_concept_scheme_concepts[s]
        self.filtered_concept_schemes.intersection_update(self.include_concept_schemes)
        return concepts_to_include

    def filter(self):
        concepts_to_exclude = set()
        concepts_to_include = set()
        if self.exclude_concept_schemes:
            concepts_to_exclude.update(self._filter_exclude_schemes())
        if self.exclude_concept_collections:
            concepts_to_exclude.update(self._filter_exclude_collections())
        if self.include_concept_schemes:
            concepts_to_include.update(self._filter_include_schemes())
        if self.include_concept_collections:
            concepts_to_include.update(self._filter_include_collections())
        overlap = concepts_to_exclude.intersection(concepts_to_include)
        if len(overlap) > 0:
            raise RuntimeError(
                f"{len(overlap)} concept/s are marked to be both excluded and included, eg:\n{next(iter(overlap))}")
        if self.exclude_concepts:
            concepts_to_exclude.update(self.exclude_concepts)
        if self.include_concepts:
            concepts_to_include.update(self.include_concepts)
        for c in concepts_to_exclude:
            if c in self.filtered_concept_maps:
                del self.filtered_concept_maps[c]

        self.filtered_concepts.difference_update(concepts_to_exclude)
        # scan every remaining concept to see if it's in the whitelist
        if concepts_to_include:
            for c in self.filtered_concepts.difference(concepts_to_include):
                if c in self.filtered_concept_maps:
                    del self.filtered_concept_maps[c]
            self.filtered_concepts.intersection_update(concepts_to_include)
        # find concept schemes to remove where all members were removed by excluded collections
        schemes_to_delete = set()
        for s in self.filtered_concept_schemes:
            if not any(c in self.filtered_concepts for c in self.filtered_concept_scheme_concepts[s]):
                del self.filtered_concept_scheme_concepts[s]
                schemes_to_delete.add(s)
        self.filtered_concept_schemes.difference_update(schemes_to_delete)

        # find concept collections to remove where all members were removed by excluded schemes
        collections_to_delete = set()
        for c in self.filtered_collections:
            if not any(m in self.filtered_concepts for m in self.filtered_collection_members[c]):
                del self.filtered_collection_members[c]
                del self.filtered_collection_maps[c]
                collections_to_delete.add(c)
        self.filtered_collections.difference_update(collections_to_delete)

    def clean_scheme(self, scheme_uri: rdflib.URIRef, scheme_cbd_graph: rdflib.Graph) -> None:
        for o in list(scheme_cbd_graph.objects(scheme_uri, SKOS.semanticRelation)):
            scheme_cbd_graph.remove((scheme_uri, SKOS.semanticRelation, o))
        for o in list(scheme_cbd_graph.objects(scheme_uri, DCTERMS.hasPart)):
            scheme_cbd_graph.remove((scheme_uri, DCTERMS.hasPart, o))
        # We add our own topConcepts using filtered broadest concepts
        for o in list(scheme_cbd_graph.objects(scheme_uri, SKOS.hasTopConcept)):
            scheme_cbd_graph.remove((scheme_uri, SKOS.hasTopConcept, o))
        return None

    def clean_collection(self, collection_uri: rdflib.URIRef, coll_cbd_graph: rdflib.Graph) -> None:
        for o in list(coll_cbd_graph.objects(collection_uri, SKOS.semanticRelation)):
            coll_cbd_graph.remove((collection_uri, SKOS.semanticRelation, o))
        for o in list(coll_cbd_graph.objects(collection_uri, DCTERMS.hasPart)):
            coll_cbd_graph.remove((collection_uri, DCTERMS.hasPart, o))
        # We add our own topConcepts using filtered broadest concepts
        for o in list(coll_cbd_graph.objects(collection_uri, SKOS.member)):
            coll_cbd_graph.remove((collection_uri, SKOS.member, o))
        return None

    def clean_concept(self, concept_uri: rdflib.URIRef, concept_cbd_graph: rdflib.Graph) -> None:
        for o in list(concept_cbd_graph.objects(concept_uri, SKOS.semanticRelation)):
            concept_cbd_graph.remove((concept_uri, SKOS.semanticRelation, o))
        # TODO: remove skos:broaderTransitive, skos:narrowerTransitive ?
        # We add our own skos:topConceptOf
        for o in list(concept_cbd_graph.objects(concept_uri, SKOS.topConceptOf)):
            concept_cbd_graph.remove((concept_uri, SKOS.topConceptOf, o))
        # We add our own skos:inScheme
        for o in list(concept_cbd_graph.objects(concept_uri, SKOS.inScheme)):
            concept_cbd_graph.remove((concept_uri, SKOS.inScheme, o))
        return None

    async def harvest(self) -> List[VocabGraphDetails]:
        if self.vocab_type == OWL.Ontology:  # This is a special TERN Ontology-ConceptScheme-Vocab, treat differently
            new_scheme_vocab_details = await self.harvest_from_ontology_vocab()
        elif self.vocab_type == SKOS.ConceptScheme:  # This is maybe a VocPub compliant Vocabulary, treat it well
            vocab_graph_detail = await self.harvest_from_concept_scheme(self.root_node)

            new_scheme_vocab_details = [vocab_graph_detail]
        else:
            new_scheme_vocab_details = []
            # Just find all concept schemes
            for scheme in self.filtered_concept_schemes:
                vocab_graph_detail = await self.harvest_from_concept_scheme(scheme)
                new_scheme_vocab_details.append(vocab_graph_detail)
        return new_scheme_vocab_details

    async def harvest_from_ontology_vocab(self) -> List[VocabGraphDetails]:
        top_concepts = set(self.root_node_details.objects(self.root_node, SKOS.hasTopConcept))
        top_concepts_2 = set(self.root_node_details.subjects(SKOS.topConceptOf, self.root_node))
        has_parts = set(
            self.root_node_details.objects(self.root_node, DCTERMS.hasPart))  # Non-standard, but TERN uses hasPart for Collections
        if len(top_concepts) < 1 and len(top_concepts_2) < 1 and len(has_parts) < 1:
            raise RuntimeError(f"No topConcepts or hasPart collections found in Vocabulary: {self.root_node}")
        defined_bys = set(self.root_node_details.subjects(RDFS.isDefinedBy, self.root_node))
        for defined_by in defined_bys:
            if defined_by in self.collection_maps:
                self.collection_maps[defined_by]['defined_by'].add(self.root_node)
            elif defined_by not in self.concept_maps:
                print(f"Skipping unknown concept is defined by Vocab Ontology: {defined_by}")
                continue
            else:
                self.concept_maps[defined_by]["defined_by"].add(self.root_node)
        use_collections = []
        part_defs = {p: await self.cbd(p) for p in has_parts if isinstance(p, rdflib.URIRef)}
        for part_uri, part_def in part_defs.items():
            part_types = set(part_def.objects(part_uri, RDF.type))
            if SKOS.Collection in part_types:
                if part_uri not in self.exclude_concept_collections:
                    use_collections.append(part_uri)
        if len(use_collections) < 1:
            raise RuntimeError(f"No topConcepts or linked collections found in Vocabulary: {self.root_node}")
        # Filter on collections and treat the ontology as a TopConcept
        old_include_collections = self.include_concept_collections
        self.include_concept_collections = use_collections
        self.filter()
        self.include_concept_collections = old_include_collections
        return [await self.harvest_from_concept_scheme(self.root_node, force_concepts=True)]


    def extract_label_from_cbd(self, target_uri: Identifier, cbd_graph: rdflib.Graph, tokenize=False) -> str:
        preflabels = set(cbd_graph.objects(target_uri, SKOS.prefLabel))
        label = None
        if len(preflabels) > 1:
            for p in preflabels:
                if isinstance(p, rdflib.Literal) and (p.language is None or p.language == "en"):
                    preflabel = p
                    break
            else:
                raise RuntimeError(f"Cannot determine which preflabel to use to identify {target_uri}")
            label = preflabel
        elif len(preflabels) == 1:
            label = str(next(iter(preflabels)))
        else:
            titles = set(cbd_graph.objects(target_uri, DCTERMS.title))
            if len(titles) > 1:
                for r in titles:
                    if isinstance(r, rdflib.Literal) and (r.language is None or r.language == "en"):
                        title = r
                        break
                else:
                    raise RuntimeError(f"Cannot determine which dcterms:title to use to identify {target_uri}")
                label = title
            elif len(titles) == 1:
                label = str(next(iter(titles)))
            else:
                rdflabels = set(cbd_graph.objects(target_uri, RDFS.label))
                if len(rdflabels) > 1:
                    for r in rdflabels:
                        if isinstance(r, rdflib.Literal) and (r.language is None or r.language == "en"):
                            rdflabel = r
                            break
                    else:
                        raise RuntimeError(f"Cannot determine which rdfs:label to use to identify {target_uri}")
                    label = rdflabel
                elif len(rdflabels) == 1:
                    label = str(next(iter(rdflabels)))
                else:
                    raise RuntimeError(f"Cannot find any skos:preflabel or rdfs:label to use to identify {target_uri}")
        if tokenize:
            token_label = str(label).lower().replace(" ", "_")
            token_len = min(len(token_label), 16)
            token_label = token_label[:token_len]
            label = token_label.rstrip("_")
        return label

    async def harvest_from_concept_scheme(self, scheme_uri: rdflib.URIRef, force_concepts=False) -> VocabGraphDetails:
        print(f"Harvesting Concept Scheme {scheme_uri}")
        try:
            concepts: Set = self.filtered_concept_scheme_concepts[scheme_uri]
        except LookupError:
            if not force_concepts:
                raise RuntimeError(f"Found no concepts for the ConceptScheme: {scheme_uri}")
            else:
                concepts = self.filtered_concepts
        if len(concepts) < 1:
            if not force_concepts:
                raise RuntimeError(f"Found no concepts for the ConceptScheme: {scheme_uri}")
            else:
                concepts = self.filtered_concepts
        scheme_graph = await self.cbd(scheme_uri)
        if scheme_graph is None or len(scheme_graph) < 1:
            raise RuntimeError(f"Cannot get CBD for ConceptScheme: {scheme_uri}")
        token = self.extract_label_from_cbd(scheme_uri, scheme_graph, tokenize=True)
        print(token)
        self.clean_scheme(scheme_uri, scheme_graph)
        top_concepts: Set = concepts.intersection(self.broadest_concepts)
        loop = asyncio.get_event_loop()
        jobs = [loop.create_task(async_task_association(self.cbd(c), c)) for c in concepts]
        await asyncio.sleep(0)  # Yield asyncio to kick-start jobs
        done_jobs = await asyncio.gather(*jobs, return_exceptions=True)
        for d in done_jobs:
            if isinstance(d, Exception):
                print(d)
                continue
            g, c = d
            self.clean_concept(c, g)
            for t in g.triples((None, None, None)):
                scheme_graph.add(t)

        # collections with concepts that are in this scheme
        applicable_collections = set(c for c in self.filtered_collections if any(m in concepts for m in self.filtered_collection_members[c]))

        # collections with members that are collections with members in this scheme
        more_applicable_collections = set(c for c in self.filtered_collections if any(m in applicable_collections for m in self.filtered_collection_members[c]))
        applicable_collections.update(more_applicable_collections)
        jobs = [loop.create_task(async_task_association(self.cbd(c), c)) for c in applicable_collections]
        await asyncio.sleep(0)  # Yield asyncio to kick-start jobs
        done_jobs = await asyncio.gather(*jobs, return_exceptions=True)
        for d in done_jobs:
            if isinstance(d, Exception):
                print(d)
                continue
            g, c = d
            self.clean_collection(c, g)
            for t in g.triples((None, None, None)):
                scheme_graph.add(t)
        for c in applicable_collections:
            scheme_graph.add((c, SKOS.inScheme, scheme_uri))
            members: Set = self.filtered_collection_members[c]
            for m in concepts.intersection(members):
                scheme_graph.add((c, SKOS.member, m))  # Concept m is member of Collection c
            for m in applicable_collections.intersection(members):
                scheme_graph.add((c, SKOS.member, m))  # Collection m is member of Collection c
        for t in top_concepts:
            scheme_graph.add((scheme_uri, SKOS.hasTopConcept, t))
            scheme_graph.add((t, SKOS.topConceptOf, scheme_uri))
        for c in concepts:
            scheme_graph.add((c, SKOS.inScheme, scheme_uri))
        return VocabGraphDetails(graph=scheme_graph,
                                 keywords=[],
                                 themes=[],
                                 token=token,
                                 vocab_uri=scheme_uri)





    async def cbd(self, identifier: rdflib.URIRef) -> rdflib.Graph:
        raise NotImplementedError()

    async def subjects(self, p, o):
        raise NotImplementedError()

    async def objects(self, s, p):
        raise NotImplementedError()

    async def get_broadest_concepts(self) -> Set[rdflib.URIRef]:
        raise NotImplementedError()

    async def get_all_concepts(self) -> Set[rdflib.URIRef]:
        raise NotImplementedError()

    async def get_concept_scheme_members(self, s: rdflib.URIRef) -> Tuple[Set[Identifier], Set[Identifier]]:
        raise NotImplementedError()

    async def get_collection_members(self, c: rdflib.URIRef) -> Set[Identifier]:
        raise NotImplementedError()


class SPARQLVocabHarvester(VocabHarvester):
    def __init__(self, source_graph: rdflib.Graph):
        super().__init__(source_graph)

    async def cbd(self, identifier: rdflib.URIRef) -> rdflib.Graph:
        return await sparql_describe(self.source_graph, identifier)

    async def subjects(self, p, o):
        return await sparql_subjects(self.source_graph, p, o)

    async def objects(self, s, p):
        return await sparql_objects(self.source_graph, s, p)

    async def get_broadest_concepts(self) -> Set[rdflib.URIRef]:
        return await sparql_broadest_concepts(self.source_graph)

    async def get_all_concepts(self) -> Set[rdflib.URIRef]:
        return await sparql_all_concepts(self.source_graph)

    async def get_concept_scheme_members(self, s: rdflib.URIRef) -> Tuple[Set[Identifier], Set[Identifier]]:
        return await sparql_concept_scheme_members(self.source_graph, s)

    async def get_collection_members(self, c: rdflib.URIRef) -> Set[Identifier]:
        return await sparql_collection_members(self.source_graph, c)

class LocalVocabHarvester(VocabHarvester):
    def __init__(self, source_graph: rdflib.Graph):
        super().__init__(source_graph)

    async def cbd(self, identifier: rdflib.URIRef) -> rdflib.Graph:
        return self.source_graph.cbd(identifier)

    async def subjects(self, p, o):
        return self.source_graph.subjects(p, o)

    async def objects(self, s, p):
        return self.source_graph.objects(s, p)

    async def get_broadest_concepts(self) -> Set[rdflib.URIRef]:
        return get_broadest_concepts(self.source_graph)

    async def get_all_concepts(self) -> Set[rdflib.URIRef]:
        return get_all_concepts(self.source_graph)

    async def get_concept_scheme_members(self, s: rdflib.URIRef) -> Tuple[Set[Identifier], Set[Identifier]]:
        return get_concept_scheme_members(self.source_graph, s)

    async def get_collection_members(self, c: rdflib.URIRef) -> Set[Identifier]:
        return get_collection_members(self.source_graph, c)


def check_type(graph: rdflib.Graph, root_node: rdflib.URIRef) -> rdflib.URIRef:
    types = set(graph.objects(root_node, RDF.type))
    is_dcat_catalog = DCAT.Catalog in types
    is_skos_conceptscheme = SKOS.ConceptScheme in types
    is_skos_collection = SKOS.Collection in types
    is_ontology = OWL.Ontology in types
    if is_skos_conceptscheme and is_skos_collection:
        raise RuntimeError("A target vocabulary cannot be both SKOS:Collection and SKOS:ConceptScheme at the same time.")
    if is_dcat_catalog and is_skos_conceptscheme:
        raise RuntimeError("A target vocabulary cannot be both DCAT:Catalog and SKOS:Collection at the same time.")
    if is_dcat_catalog and is_skos_collection:
        raise RuntimeError("A target vocabulary cannot be both DCAT:Catalog and SKOS:ConceptScheme at the same time.")
    if is_dcat_catalog:
        raise NotImplementedError("Target vocabulary in a DCAT:Catalog not yet implemented.")
    if is_skos_collection:
        raise NotImplementedError("Target vocabulary as a SKOS:Collection not yet implemented.")
    if not is_skos_conceptscheme:
        raise RuntimeError("Target vocabulary is not a SKOS:ConceptScheme. Double check root_node.")
    if is_ontology:
        return OWL.Ontology  # This implies both OWL:Ontology, and ConceptScheme, as so used in TERN
    else:
        return SKOS.ConceptScheme



