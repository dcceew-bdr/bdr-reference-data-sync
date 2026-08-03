import unittest

import rdflib
from rdflib.namespace import RDFS, SKOS

from src.harvesters.vocabularies import VocabHarvester
from src.catalog import normalise_tern_scheme_memberships
from src.voc_graph import VocabGraphDetails


class CanonicalSchemeMembershipTest(unittest.TestCase):
    def test_canonical_scheme_defines_harvested_concept(self) -> None:
        graph = rdflib.Graph()
        concept = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/example")
        canonical_scheme = rdflib.URIRef(
            "http://linked.data.gov.au/def/tern-cv/canonical"
        )
        secondary_scheme = rdflib.URIRef(
            "http://linked.data.gov.au/def/tern-cv/secondary"
        )
        harvester = VocabHarvester.__new__(VocabHarvester)
        harvester.concept_maps = {
            concept: {"concept_schemes": {canonical_scheme, secondary_scheme}}
        }

        harvester.add_canonical_scheme_membership(graph, {concept}, canonical_scheme)

        self.assertIn((concept, SKOS.inScheme, canonical_scheme), graph)
        self.assertIn((concept, RDFS.isDefinedBy, canonical_scheme), graph)
        self.assertNotIn((concept, RDFS.isDefinedBy, secondary_scheme), graph)

    def test_single_scheme_concept_does_not_gain_is_defined_by(self) -> None:
        graph = rdflib.Graph()
        concept = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/example")
        scheme = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/scheme")
        harvester = VocabHarvester.__new__(VocabHarvester)
        harvester.concept_maps = {concept: {"concept_schemes": {scheme}}}

        harvester.add_canonical_scheme_membership(graph, {concept}, scheme)

        self.assertIn((concept, SKOS.inScheme, scheme), graph)
        self.assertNotIn((concept, RDFS.isDefinedBy, scheme), graph)


class UniqueTopConceptSchemeTest(unittest.TestCase):
    tern_namespace = "http://linked.data.gov.au/def/tern-cv/"

    def make_harvester(
        self,
        concept: rdflib.URIRef,
        schemes: set[rdflib.URIRef],
        top_schemes: set[rdflib.URIRef],
    ) -> VocabHarvester:
        harvester = VocabHarvester.__new__(VocabHarvester)
        harvester.concept_maps = {concept: {"concept_schemes": schemes}}
        harvester.concept_scheme_top_concepts = {
            scheme: ({concept} if scheme in top_schemes else set())
            for scheme in schemes
        }
        harvester.filtered_concept_scheme_concepts = {
            scheme: {concept} for scheme in schemes
        }
        return harvester

    def test_unique_top_concept_scheme_is_preferred(self) -> None:
        concept = rdflib.URIRef(f"{self.tern_namespace}concept")
        canonical = rdflib.URIRef(f"{self.tern_namespace}canonical")
        secondary = rdflib.URIRef(f"{self.tern_namespace}secondary")
        harvester = self.make_harvester(
            concept, {canonical, secondary}, {canonical}
        )

        harvester.prefer_unique_top_concept_scheme()

        self.assertIn(concept, harvester.filtered_concept_scheme_concepts[canonical])
        self.assertNotIn(
            concept, harvester.filtered_concept_scheme_concepts[secondary]
        )

    def test_multiple_top_concept_schemes_are_unchanged(self) -> None:
        concept = rdflib.URIRef(f"{self.tern_namespace}concept")
        first = rdflib.URIRef(f"{self.tern_namespace}first")
        second = rdflib.URIRef(f"{self.tern_namespace}second")
        harvester = self.make_harvester(concept, {first, second}, {first, second})

        harvester.prefer_unique_top_concept_scheme()

        self.assertIn(concept, harvester.filtered_concept_scheme_concepts[first])
        self.assertIn(concept, harvester.filtered_concept_scheme_concepts[second])

    def test_no_top_concept_scheme_is_unchanged(self) -> None:
        concept = rdflib.URIRef(f"{self.tern_namespace}concept")
        first = rdflib.URIRef(f"{self.tern_namespace}first")
        second = rdflib.URIRef(f"{self.tern_namespace}second")
        harvester = self.make_harvester(concept, {first, second}, set())

        harvester.prefer_unique_top_concept_scheme()

        self.assertIn(concept, harvester.filtered_concept_scheme_concepts[first])
        self.assertIn(concept, harvester.filtered_concept_scheme_concepts[second])

    def test_mixed_tern_and_non_tern_schemes_are_unchanged(self) -> None:
        concept = rdflib.URIRef(f"{self.tern_namespace}concept")
        tern_scheme = rdflib.URIRef(f"{self.tern_namespace}tern-scheme")
        external_scheme = rdflib.URIRef("https://example.com/external-scheme")
        harvester = self.make_harvester(
            concept, {tern_scheme, external_scheme}, {tern_scheme}
        )

        harvester.prefer_unique_top_concept_scheme()

        self.assertIn(
            concept, harvester.filtered_concept_scheme_concepts[tern_scheme]
        )
        self.assertIn(
            concept, harvester.filtered_concept_scheme_concepts[external_scheme]
        )


class FinalTernSchemeNormalisationTest(unittest.TestCase):
    def make_detail(self, scheme: rdflib.URIRef, graph: rdflib.Graph) -> VocabGraphDetails:
        return VocabGraphDetails(
            graph=graph, keywords=[], themes=[], token=str(scheme), vocab_uri=scheme
        )

    def test_adjacent_describe_membership_is_removed_from_secondary_graph(self) -> None:
        concept = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/concept")
        canonical = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/canonical")
        secondary = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/secondary")
        canonical_graph = rdflib.Graph()
        secondary_graph = rdflib.Graph()
        canonical_graph.add((concept, SKOS.inScheme, canonical))
        canonical_graph.add((concept, SKOS.topConceptOf, canonical))
        secondary_graph.add((concept, SKOS.inScheme, secondary))

        normalise_tern_scheme_memberships([
            self.make_detail(canonical, canonical_graph),
            self.make_detail(secondary, secondary_graph),
        ])

        combined = canonical_graph + secondary_graph
        self.assertEqual(set(combined.objects(concept, SKOS.inScheme)), {canonical})
        self.assertIn((concept, RDFS.isDefinedBy, canonical), combined)

    def test_existing_secondary_is_defined_by_is_preserved(self) -> None:
        concept = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/concept")
        canonical = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/canonical")
        secondary = rdflib.URIRef("http://linked.data.gov.au/def/tern-cv/secondary")
        external_definition = rdflib.URIRef("https://example.com/source")
        canonical_graph = rdflib.Graph()
        secondary_graph = rdflib.Graph()
        canonical_graph.add((concept, SKOS.inScheme, canonical))
        canonical_graph.add((concept, SKOS.topConceptOf, canonical))
        secondary_graph.add((concept, SKOS.inScheme, secondary))
        secondary_graph.add((concept, RDFS.isDefinedBy, external_definition))

        normalise_tern_scheme_memberships([
            self.make_detail(canonical, canonical_graph),
            self.make_detail(secondary, secondary_graph),
        ])

        self.assertIn((concept, RDFS.isDefinedBy, external_definition), secondary_graph)


if __name__ == "__main__":
    unittest.main()
