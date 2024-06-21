from typing import List

import rdflib
from dataclasses import dataclass
from rdflib.namespace import DCAT, SKOS, DCTERMS
TERN = rdflib.Namespace("https://w3id.org/tern/ontologies/tern/")


@dataclass
class VocabGraphDetails:
    graph: rdflib.Graph
    keywords: List[str]
    themes: List[rdflib.URIRef]
    token: str
    vocab_uri: rdflib.URIRef  # This is the ConceptScheme


def make_voc_graph():
    g = rdflib.Graph(bind_namespaces="core")
    ns = g.namespace_manager
    ns.bind("skos", SKOS)
    ns.bind("tern", TERN)
    ns.bind("dcat", DCAT)
    ns.bind("dcterms", DCTERMS)
    return g
