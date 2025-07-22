from typing import List, Optional

import rdflib
from dataclasses import dataclass

from rdflib.namespace import DCAT, SKOS, DCTERMS, VANN, SOSA, SDO
TERN = rdflib.Namespace("https://w3id.org/tern/ontologies/tern/")
ABIS = rdflib.Namespace("https://linked.data.gov.au/def/abis/")
BDRDS = rdflib.Namespace("https://linked.data.gov.au/dataset/bdr/")
BDRPR = rdflib.Namespace("https://linked.data.gov.au/def/bdr-pr/")


@dataclass
class VocabGraphDetails:
    graph: rdflib.Graph
    keywords: List[str]
    themes: List[str]
    token: str
    vocab_uri: rdflib.URIRef  # This is the SKOS:ConceptScheme
    graph_name: Optional[rdflib.URIRef] = None

@dataclass
class CatalogGraphDetails:
    graph: rdflib.Graph
    token: str
    cat_uri: rdflib.URIRef  # This is the DCAT:Catalog
    content_graphs: List[VocabGraphDetails]
    graph_name: Optional[rdflib.URIRef] = None


def make_voc_graph(multigraph: bool = False):
    g = rdflib.Graph(bind_namespaces="core")
    ns = g.namespace_manager
    ns.bind("skos", SKOS)
    ns.bind("sosa", SOSA)
    ns.bind("tern", TERN)
    ns.bind("dcat", DCAT)
    ns.bind("dcterms", DCTERMS)
    ns.bind("vann", VANN)
    ns.bind("schema", SDO)
    ns.bind("abis", ABIS)
    ns.bind("bdr-ds", BDRDS)
    ns.bind("bdr-pr", BDRPR)
    if multigraph:
        ds = rdflib.Dataset(store=g.store)
        ds.namespace_manager = ns
        return ds
    return g
