import urllib.parse
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import httpx
import rdflib
from rdflib.namespace import OWL
from rdflib.plugins.stores import sparqlstore

from .harvesters.sparql_fetch import remote_sparql, remote_sparql_update


EXPLICIT_GRAPH = rdflib.URIRef("http://www.ontotext.com/explicit")


@dataclass
class SelfSameAsCleanupResult:
    token: str
    query_endpoint: str
    update_endpoint: str
    subjects: list[rdflib.URIRef]
    applied: bool


def iter_graphdb_catalogues(catalog_defs: Iterable[dict[str, Any]], tokens: Optional[set[str]] = None):
    for catalog_def in catalog_defs:
        token = catalog_def.get("token", catalog_def.get("name", "unnamed"))
        if tokens is not None and token not in tokens:
            continue
        source = catalog_def.get("source", "")
        if not source.lower().startswith("sparql:"):
            continue
        if not bool(catalog_def.get("is_graph_db", False)):
            continue
        yield catalog_def


def sparql_query_endpoint(source: str) -> str:
    if not source.lower().startswith("sparql:"):
        raise ValueError(f"Not a SPARQL source: {source}")
    return source[7:]


def graphdb_update_endpoint(query_endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(query_endpoint)
    if parsed.path.endswith("/statements"):
        return query_endpoint
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/") + "/statements", parsed.query, parsed.fragment)
    )


def make_graphdb_source_graph(query_endpoint: str) -> rdflib.Graph:
    store = sparqlstore.SPARQLStore(query_endpoint=query_endpoint, method="POST", returnFormat="json")
    return rdflib.Graph(store=store, bind_namespaces="core")


async def find_self_referential_same_as(query_endpoint: str) -> list[rdflib.URIRef]:
    graph = make_graphdb_source_graph(query_endpoint)
    query = f"""\
PREFIX owl: <{OWL}>
SELECT DISTINCT ?s
FROM <{EXPLICIT_GRAPH}>
WHERE {{
  ?s owl:sameAs ?s .
}}
ORDER BY ?s
"""
    results = await remote_sparql(graph, query, infer=False)
    return [row["s"] for row in results if isinstance(row["s"], rdflib.URIRef)]


async def delete_self_referential_same_as(update_endpoint: str) -> None:
    default_graph_update = f"""\
PREFIX owl: <{OWL}>
DELETE WHERE {{
  ?s owl:sameAs ?s .
}}
"""
    named_graph_update = f"""\
PREFIX owl: <{OWL}>
DELETE {{
  GRAPH ?g {{
    ?s owl:sameAs ?s .
  }}
}}
WHERE {{
  GRAPH ?g {{
    ?s owl:sameAs ?s .
  }}
}}
"""
    async with httpx.AsyncClient(timeout=(30.0, 60.0, 6.0, 60.0)) as client:
        await remote_sparql_update(update_endpoint, default_graph_update, client=client)
        await remote_sparql_update(update_endpoint, named_graph_update, client=client)


async def cleanup_catalogue_self_same_as(catalog_def: dict[str, Any], apply: bool = False) -> SelfSameAsCleanupResult:
    token = catalog_def.get("token", catalog_def.get("name", "unnamed"))
    query_endpoint = sparql_query_endpoint(catalog_def["source"])
    update_endpoint = catalog_def.get("update_endpoint", graphdb_update_endpoint(query_endpoint))
    subjects = await find_self_referential_same_as(query_endpoint)
    if apply and len(subjects) > 0:
        await delete_self_referential_same_as(update_endpoint)
    return SelfSameAsCleanupResult(
        token=token,
        query_endpoint=query_endpoint,
        update_endpoint=update_endpoint,
        subjects=subjects,
        applied=apply and len(subjects) > 0,
    )
