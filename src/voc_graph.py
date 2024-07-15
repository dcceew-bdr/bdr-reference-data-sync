from typing import List

import rdflib
from dataclasses import dataclass

from rdflib.namespace import DCAT, SKOS, DCTERMS, VANN, SOSA, SDO
TERN = rdflib.Namespace("https://w3id.org/tern/ontologies/tern/")
ABIS = rdflib.Namespace("https://linked.data.gov.au/def/abis/")
BDR = rdflib.Namespace("https://linked.data.gov.au/dataset/bdr/")


@dataclass
class VocabGraphDetails:
    graph: rdflib.Graph
    keywords: List[str]
    themes: List[rdflib.URIRef]
    token: str
    vocab_uri: rdflib.URIRef  # This is the SKOS:ConceptScheme


@dataclass
class CatalogGraphDetails:
    graph: rdflib.Graph
    token: str
    cat_uri: rdflib.URIRef  # This is the DCAT:Catalog
    content_graphs: List[VocabGraphDetails]


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
    ns.bind("bdr", BDR)
    if multigraph:
        ds = rdflib.Dataset(store=g.store)
        ds.namespace_manager = ns
        return ds
    return g
