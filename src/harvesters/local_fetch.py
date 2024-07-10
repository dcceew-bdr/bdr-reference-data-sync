from typing import Set, Tuple

import rdflib
from rdflib import SKOS, RDF
from rdflib.term import Identifier

from src.voc_graph import TERN


def get_all_concepts(
    graph: rdflib.Graph,
) -> Set[rdflib.URIRef]:
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    SELECT DISTINCT ?c WHERE
    {{
      {{ ?c rdf:type skos:Concept }} UNION
      {{
        {{ ?a skos:hasTopConcept ?c }} UNION {{ ?c skos:topConceptOf ?b }}
      }} UNION {{
        {{
          {{ ?d skos:narrower ?c }} UNION {{ ?e skos:broader ?c }}
        }} UNION {{
          {{ ?c skos:narrower ?f }} UNION {{ ?c skos:broader ?g }}
        }}
      }} UNION {{
        ?c skos:inScheme ?s
      }} UNION {{
        ?c skos:definition ?h
      }}
      FILTER NOT EXISTS {{
        {{ ?c rdf:type skos:Collection }} UNION {{ ?c rdf:type skos:OrderedCollection }} UNION {{ ?c rdf:type skos:ConceptScheme }}
      }}
    }}\
    '''
    sparql_res = graph.query(sparql, initNs={})
    if len(sparql_res) < 1:
        return set()
    return set(r['c'] for r in sparql_res)


def get_broadest_concepts(
    graph: rdflib.Graph,
) -> Set[rdflib.URIRef]:
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    SELECT DISTINCT ?c WHERE
    {{
      {{
        ?c rdf:type skos:Concept
      }} UNION {{
        {{
          {{ ?a skos:hasTopConcept ?c }} UNION {{ ?c skos:topConceptOf ?b }}
        }} UNION {{
          {{ ?c skos:narrower ?d }} UNION {{ ?e skos:broader ?c }}
        }}
      }}
      FILTER NOT EXISTS {{
        {{ ?x skos:narrower ?c }} UNION {{ ?c skos:broader ?x }}
      }}
    }}\
    '''
    sparql_res = graph.query(sparql, initNs={})
    if len(sparql_res) < 1:
        return set()
    return set(r['c'] for r in sparql_res)


def get_concept_scheme_hierarchy(
    graph: rdflib.Graph,
    s: rdflib.URIRef,
) -> Tuple[Set[Identifier], Set[Identifier]]:
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    SELECT DISTINCT ?t ?n WHERE
    {{ BIND (<{s}> as ?s).
      {{ ?s skos:hasTopConcept ?t }} UNION {{ ?t skos:topConceptOf ?s }}
      OPTIONAL {{ {{ ?t skos:narrower+ ?n }} UNION {{ ?n skos:broader+ ?t }} }}
    }}\
    '''
    sparql_res = graph.query(sparql, initNs={})
    if len(sparql_res) < 1:
        return (set(), set())
    narrowers = set()
    tops = set()
    for r in sparql_res:
        tops.add(r['t'])
        if r['n'] is not None:
            narrowers.add(r['n'])
    return (tops, narrowers)


def get_concept_scheme_concepts(
    graph: rdflib.Graph,
    s: rdflib.URIRef,
) -> Set[Identifier]:
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    SELECT DISTINCT ?c WHERE
    {{ BIND (<{s}> as ?s).
      {{ ?c skos:inScheme ?s }}
    }}\
    '''
    sparql_res = graph.query(sparql, initNs={})
    if len(sparql_res) < 1:
        return set()
    concepts = set()
    for r in sparql_res:
        concepts.add(r['c'])
    return concepts


def get_collection_immediate_members(
    graph: rdflib.Graph,
    c: rdflib.URIRef
) -> Set[Identifier]:
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    PREFIX tern: <{str(TERN)}>
    SELECT DISTINCT ?m WHERE
    {{ BIND (<{c}> as ?c).
      {{ {{ ?c skos:member ?m }} UNION {{ ?m tern:isMemberOf ?c }} }} UNION {{ ?m tern:hasCategoricalCollection ?c }}
    }}\
    '''
    sparql_res = graph.query(sparql, initNs={})
    if len(sparql_res) < 1:
        return set()

    return set(r['m'] for r in sparql_res)


def get_collection_all_members(
    graph: rdflib.Graph,
    c: rdflib.URIRef
) -> Set[Identifier]:
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    PREFIX tern: <{str(TERN)}>
    SELECT DISTINCT ?m WHERE
    {{ BIND (<{c}> as ?c).
      {{ {{ ?c skos:member+ ?m }} UNION {{ ?m tern:isMemberOf+ ?c }} }} UNION {{ ?m tern:hasCategoricalCollection+ ?c }}
    }}\
    '''
    sparql_res = graph.query(sparql, initNs={})
    if len(sparql_res) < 1:
        return set()

    return set(r['m'] for r in sparql_res)
