from typing import List, Dict, Any, Union, Tuple, Optional
import rdflib
from rdflib import RDF, DCAT, RDFS, VANN, XSD, SDO, DCTERMS
from rdflib.plugins.stores import sparqlstore
from pathlib import Path
from .harvesters import VocabHarvester
from .voc_graph import make_voc_graph, VocabGraphDetails, CatalogGraphDetails

bdr_cat_ns = rdflib.Literal("https://linked.data.gov.au/dataset/bdr/catalogs/", datatype=XSD.anyURI)

async def build_catalog(catalog_def: Dict[str, Any], serialize=True) -> CatalogGraphDetails:
    try:
        cat_source = catalog_def["source"]
    except LookupError:
        raise RuntimeError("No source defined on catalog definition.")

    harvesters = {}
    if cat_source is not None and len(cat_source) > 0:
        harvesters[cat_source] = cat_harvester = VocabHarvester.build_from_source(cat_source, catalog_def)
    else:
        cat_harvester = None
    cat_token = catalog_def["token"]
    cat_graph_name: Optional[str] = catalog_def.get("graph_name", None)
    cat_path = Path(".") / "generated" / cat_token
    cat_path.mkdir(exist_ok=True, parents=True)
    cat_graph = make_voc_graph()
    cat_uri = rdflib.URIRef(f"https://linked.data.gov.au/dataset/bdr/catalogs/{cat_token}")
    if cat_graph_name is not None:
        cat_graph_uri = rdflib.URIRef(cat_graph_name)
    else:
        cat_graph_uri = rdflib.URIRef(cat_uri + "-catalogue")
    cat_graph.add((cat_uri, RDF.type, DCAT.Catalog))
    cat_graph.add((cat_uri, VANN.preferredNamespacePrefix, rdflib.Literal("bdr-cat")))
    cat_graph.add((cat_uri, VANN.preferredNamespaceUri, bdr_cat_ns))
    cat_graph.add((cat_uri, DCAT.themeTaxonomy, rdflib.URIRef("https://linked.data.gov.au/def/abis/vocab-themes")))
    cat_graph.add((cat_uri, DCTERMS.title, rdflib.Literal(catalog_def.get("label"))))
    vocabularies: List[Dict] = catalog_def.get("vocabularies", [])
    vocab_graph_details: List[VocabGraphDetails] = []
    for vocab_def in vocabularies:
        vocab_source = vocab_def.get("source", None)
        if vocab_source is not None and len(vocab_source) > 0:
            if vocab_source in harvesters:
                harvester = harvesters[vocab_source]
            else:
                harvester = VocabHarvester.build_from_source(vocab_source, vocab_def)
        else:
            if cat_harvester is None:
                raise RuntimeError(
                    "Cannot find a harvester for the catalog source, and none was specified in Vocab Definition")
            harvester = cat_harvester
        harvester.load_def(vocab_def)
        these_vocab_graphs_details: List[VocabGraphDetails] = await harvester.run_procedures()
        vocab_graph_details.extend(these_vocab_graphs_details)
    namespaces = catalog_def.get("namespaces", [])
    for namespace_def in namespaces:
        namespace_name = str(namespace_def.get("name", "unnamed"))
        namespace_vann_prefix = str(namespace_def.get("vann_prefix", ""))
        namespace_vann_namespace = str(namespace_def.get("vann_namespace", ""))
        if len(namespace_vann_prefix) == 0 or len(namespace_vann_namespace) == 0:
            raise RuntimeError(f"Namespace {namespace_name} is missing a vann_prefix or vann_namespace")
        str(namespace_vann_namespace)
        ns_def_uri = rdflib.URIRef(f"https://linked.data.gov.au/dataset/bdr/ns/{namespace_name}")
        cat_graph.add((ns_def_uri, VANN.preferredNamespacePrefix, rdflib.Literal(namespace_vann_prefix)))
        cat_graph.add((ns_def_uri, VANN.preferredNamespaceUri, rdflib.Literal(namespace_vann_namespace, datatype=XSD.anyURI)))
        cat_graph.add((cat_uri, DCTERMS.hasPart, ns_def_uri))  # TODO: <-- Is this needed?
        cat_graph.add((ns_def_uri, DCTERMS.isPartOf, cat_uri))  # TODO: <-- Is this needed?
    cat_details = CatalogGraphDetails(
        graph=cat_graph,
        token=cat_token,
        cat_uri=cat_uri,
        content_graphs=vocab_graph_details,
        graph_name=cat_graph_uri,
    )
    for vocab_graph_detail in vocab_graph_details:
        vocab_uri = vocab_graph_detail.vocab_uri
        cat_graph.add((vocab_uri, RDF.type, DCAT.Dataset))
        keywords: List[str] = vocab_graph_detail.keywords
        if len(keywords) > 0:
            cat_graph.add((vocab_uri, SDO.keywords, rdflib.Literal(", ".join(keywords))))
        themes: List[str] = vocab_graph_detail.themes
        for t in themes:
            cat_graph.add((vocab_uri, DCAT.theme, rdflib.URIRef(t)))
        cat_graph.add((cat_uri, DCTERMS.hasPart, vocab_uri))
        cat_graph.add((vocab_uri, DCTERMS.isPartOf, cat_uri))
        if serialize:
            out_dir = cat_path / "vocabularies"
            out_dir.mkdir(exist_ok=True, parents=True)
            out_file = out_dir / f"{vocab_graph_detail.token}.ttl"
            with open(out_file, "wb") as f:
                vocab_graph_detail.graph.serialize(f, format="turtle")

    if serialize:
        cat_file = cat_path / "catalog.ttl"
        with open(cat_file, "wb") as f:
            cat_graph.serialize(f, format="turtle")
    return cat_details


