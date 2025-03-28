import json
import urllib
import urllib.parse
from typing import Union, List, Dict, Any, Tuple, Set
import threading
import httpx
import rdflib
from rdflib.plugins.sparql.results import jsonresults
from rdflib.term import Identifier  # identifier includes URIRef, BNode, and Literal
from rdflib.namespace import SKOS

from ..voc_graph import make_voc_graph, TERN


def get_httpx_client() -> httpx.AsyncClient:
    try:
        cache = get_httpx_client.cache
    except (AttributeError, LookupError):
        get_httpx_client.cache = cache = threading.local()
    try:
        client = cache.client
    except (AttributeError, LookupError):
        # timeouts are (connect_timeout, read_timeout, write_timeout, pool_timeout)
        cache.client = client = httpx.AsyncClient(
            timeout=(30.0, 60.0, 6.0, 60.0),
            limits=httpx.Limits(max_connections=4)
        )
    return client
get_httpx_client.cache = threading.local()


async def sparql_describe(
    graph: rdflib.Graph,
    identifier: rdflib.URIRef,
    client: Union[httpx.AsyncClient, None] = None,
    explicit: bool = False,  # Only list real triples from GraphDB
    infer: Union[bool, None] = False
) -> rdflib.Graph:
    # This is the same as graph.cbd but it can be run on a remote SPARQL endpoint
    explicit_clause = "FROM <http://www.ontotext.com/explicit>" if explicit else ""
    sparql = f'''DESCRIBE ?i {explicit_clause} WHERE {{ BIND (<{identifier}> as ?i). }}'''
    if client is None:
        client = get_httpx_client()
    try:
        endpoint = graph.store.query_endpoint
    except AttributeError:
        raise RuntimeError("sparql_describe only works on SPARQLStore endpoints")
    additional_args = {}
    if infer is not None:
        if infer is True:
            infer_string = "infer=true"
            additional_args["infer"] = "true"
        else:
            infer_string = "infer=false"
            additional_args["infer"] = "false"
        scheme, netloc, _url, _query_string, fragment = urllib.parse.urlsplit(endpoint)
        if len(_query_string) < 1:
            _query_string = infer_string
        else:
            _query_string = _query_string + "&"+ infer_string
        endpoint = urllib.parse.urlunsplit((scheme, netloc, _url, _query_string, fragment))
    resp = await client.post(
        endpoint, data={"query": sparql, **additional_args}, headers={"Accept": "text/turtle", "Accept-Encoding": "gzip, deflate"}
    )
    content = await resp.aread()
    g = make_voc_graph()
    g.parse(data=content, format="turtle")
    return g


async def remote_sparql(graph: rdflib.Graph, query, client: Union[httpx.AsyncClient, None] = None, infer: Union[bool, None] = False) -> jsonresults.JSONResult:
    if client is None:
        client = get_httpx_client()
    try:
        endpoint = graph.store.query_endpoint
    except AttributeError:
        raise RuntimeError("SPARQL Remote lookup only works on SPARQLStore endpoints")
    additional_args = {}
    if infer is not None:
        if infer is True:
            infer_string = "infer=true"
            additional_args["infer"] = "true"
        else:
            infer_string = "infer=false"
            additional_args["infer"] = "false"
        scheme, netloc, _url, _query_string, fragment = urllib.parse.urlsplit(endpoint)
        if len(_query_string) < 1:
            _query_string = infer_string
        else:
            _query_string = _query_string + "&"+ infer_string
        endpoint = urllib.parse.urlunsplit((scheme, netloc, _url, _query_string, fragment))
    resp = await client.post(
        endpoint, data={"query": query, **additional_args}, headers={"Accept": "application/sparql-results+json", "Accept-Encoding": "gzip, deflate"}
    )
    content = await resp.aread()
    try:
        content_dict = json.loads(content)
    except Exception as e:
        print(e)
        raise RuntimeError("Cannot parse SPARQL response from remote query.")
    return jsonresults.JSONResult(content_dict)


async def sparql_subjects(
    graph: rdflib.Graph,
    p: rdflib.URIRef,
    o: rdflib.URIRef,
    client: Union[httpx.AsyncClient, None] = None,
    explicit: bool = False,
) -> List[Identifier]:
    explicit_clause = "FROM <http://www.ontotext.com/explicit>" if explicit else ""
    sparql = f'''SELECT ?s {explicit_clause} WHERE {{ BIND (<{p}> as ?p). BIND (<{o}> as ?o). ?s ?p ?o. }}'''
    sparql_res = await remote_sparql(graph, sparql, client=client)
    if len(sparql_res) < 1:
        return []
    return [r['s'] for r in sparql_res]


async def sparql_objects(
    graph: rdflib.Graph,
    s: rdflib.URIRef,
    p: rdflib.URIRef,
    client: Union[httpx.AsyncClient, None] = None,
    explicit: bool = False,
) -> List[Identifier]:
    explicit_clause = "FROM <http://www.ontotext.com/explicit>" if explicit else ""
    sparql = f'''SELECT ?o {explicit_clause} WHERE {{ BIND (<{s}> as ?s). BIND (<{p}> as ?p). ?s ?p ?o. }}'''
    sparql_res = await remote_sparql(graph, sparql, client=client)
    if len(sparql_res) < 1:
        return []
    return [r['o'] for r in sparql_res]


async def sparql_concept_scheme_hierarchy(
    graph: rdflib.Graph,
    s: rdflib.URIRef,
    client: Union[httpx.AsyncClient, None] = None,
    explicit: bool = False,
) -> Tuple[Set[Identifier], Set[Identifier]]:
    explicit_clause = "FROM <http://www.ontotext.com/explicit>" if explicit else ""
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    SELECT DISTINCT ?t ?n {explicit_clause} WHERE
    {{ BIND (<{s}> as ?s).
      {{ ?s skos:hasTopConcept ?t }} UNION {{ ?t skos:topConceptOf ?s }}
      OPTIONAL {{ {{ ?t skos:narrower+ ?n }} UNION {{ ?n skos:broader+ ?t }} }}
    }}\
    '''
    sparql_res = await remote_sparql(graph, sparql, client=client)
    if len(sparql_res) < 1:
        return (set(), set())
    narrowers = set()
    tops = set()
    for r in sparql_res:
        tops.add(r['t'])
        if r['n'] is not None:
            narrowers.add(r['n'])
    return (tops, narrowers)


async def sparql_concept_scheme_concepts(
    graph: rdflib.Graph,
    s: rdflib.URIRef,
    client: Union[httpx.AsyncClient, None] = None,
    explicit: bool = False,
) -> Set[Identifier]:
    explicit_clause = "FROM <http://www.ontotext.com/explicit>" if explicit else ""
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    SELECT DISTINCT ?c {explicit_clause} WHERE
    {{ BIND (<{s}> as ?s).
      {{ ?c skos:inScheme ?s }}
    }}\
    '''
    sparql_res = await remote_sparql(graph, sparql, client=client)
    if len(sparql_res) < 1:
        return set()
    concepts = set()
    for r in sparql_res:
        concepts.add(r['c'])
    return concepts


async def sparql_collection_all_members(
    graph: rdflib.Graph,
    c: rdflib.URIRef,
    client: Union[httpx.AsyncClient, None] = None,
    explicit: bool = False,
) -> Set[Identifier]:
    explicit_clause = "FROM <http://www.ontotext.com/explicit>" if explicit else ""
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    PREFIX tern: <{str(TERN)}>
    SELECT DISTINCT ?m {explicit_clause} WHERE
    {{ BIND (<{c}> as ?c).
      {{ {{ ?c skos:member+ ?m }} UNION {{ ?m tern:isMemberOf+ ?c }} }} UNION {{ ?m tern:hasCategoricalCollection+ ?c }}
    }}\
    '''
    sparql_res = await remote_sparql(graph, sparql, client=client)
    if len(sparql_res) < 1:
        return set()

    return set(r['m'] for r in sparql_res)


async def sparql_collection_immediate_members(
    graph: rdflib.Graph,
    c: rdflib.URIRef,
    client: Union[httpx.AsyncClient, None] = None,
    explicit: bool = False,
) -> Set[Identifier]:
    explicit_clause = "FROM <http://www.ontotext.com/explicit>" if explicit else ""
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    PREFIX tern: <{str(TERN)}>
    SELECT DISTINCT ?m {explicit_clause} WHERE
    {{ BIND (<{c}> as ?c).
      {{ {{ ?c skos:member ?m }} UNION {{ ?m tern:isMemberOf ?c }} }} UNION {{ ?m tern:hasCategoricalCollection ?c }}
    }}\
    '''
    sparql_res = await remote_sparql(graph, sparql, client=client)
    if len(sparql_res) < 1:
        return set()

    return set(r['m'] for r in sparql_res)


async def sparql_all_concepts(
    graph: rdflib.Graph,
    client: Union[httpx.AsyncClient, None] = None,
    explicit: bool = False,
) -> Set[rdflib.URIRef]:
    explicit_clause = "FROM <http://www.ontotext.com/explicit>" if explicit else ""
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    SELECT DISTINCT ?c {explicit_clause} WHERE
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
    sparql_res = await remote_sparql(graph, sparql, client=client)
    if len(sparql_res) < 1:
        return set()
    return set(r['c'] for r in sparql_res)


async def sparql_broadest_concepts(
    graph: rdflib.Graph,
    client: Union[httpx.AsyncClient, None] = None,
    explicit: bool = False,
) -> Set[rdflib.URIRef]:
    explicit_clause = "FROM <http://www.ontotext.com/explicit>" if explicit else ""
    sparql = f'''\
    PREFIX skos: <{str(SKOS)}>
    SELECT DISTINCT ?c {explicit_clause} WHERE
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
    sparql_res = await remote_sparql(graph, sparql, client=client)
    if len(sparql_res) < 1:
        return set()

    return set(r['c'] for r in sparql_res)