
if __name__ == '__main__':
    import sys
    print("Run with python main.py or python -m src")
    sys.exit(1)

from rdflib.term import URIRef, Literal
from rdflib.namespace import Namespace, RDF
from rdflib.namespace import SDO as SCHEMA
from typing import Optional
from . import config
from .catalog import build_catalog
from .graphdb_maintenance import cleanup_catalogue_self_same_as, iter_graphdb_catalogues
from .voc_graph import make_voc_graph

# Customisable constants
ALL_VOCABS_VG = URIRef("urn:bdr:vg:all-vocabularies")
ALL_CATALOGUES_VG = URIRef("urn:bdr:vg:all-catalogues")
VOCABS_IN_INDIVIDUAL_RG = False

# Non-customisable constants
OLIS = Namespace("https://olis.dev/")
SystemGraphURI = OLIS.system

async def build_catalogues(catalogue_tokens: Optional[list[str]] = None):
    catalog_defs = config.get_value("catalogues", None)
    if catalog_defs is None or len(catalog_defs) == 0:
        raise Exception("No catalogues defined")

    token_filter = set(catalogue_tokens) if catalogue_tokens else None
    selected_catalogues = [
        catalog_def for catalog_def in catalog_defs
        if token_filter is None or catalog_def.get("token") in token_filter
    ]
    if token_filter:
        missing_tokens = token_filter - {catalog_def.get("token") for catalog_def in selected_catalogues}
        if missing_tokens:
            raise RuntimeError("Unknown catalogue token(s): " + ", ".join(sorted(missing_tokens)))

    for catalog_def in selected_catalogues:
        if 'token' not in catalog_def:
            raise RuntimeError("Catalogue entry does not have token property.")
        print(f"Building Catalogue: {catalog_def['token']}", flush=True)
        cat_details = await build_catalog(catalog_def)
        cat_ds = make_voc_graph(multigraph=True)
        cat_vg_uri: URIRef = cat_details.cat_uri
        cat_rg_uri = cat_details.graph_name
        if cat_rg_uri is None:
            # Fall-back to auto-generated, because this cannot be the same as the catalogue name
            cat_rg_uri = URIRef(str(cat_vg_uri).rstrip("/#") + "-cat-rg")
        if cat_vg_uri == cat_rg_uri:
            raise RuntimeError(f"Catalogue real-graph and virtual-graph URIs cannot be the same.\n<{cat_vg_uri}>==<{cat_rg_uri}>")
        for (s, p, o) in cat_details.graph:
            cat_ds.add((s, p, o, cat_rg_uri))
        cat_ds.add((ALL_VOCABS_VG, RDF.type, OLIS.VirtualGraph, SystemGraphURI))
        cat_ds.add((ALL_CATALOGUES_VG, RDF.type, OLIS.VirtualGraph, SystemGraphURI))
        cat_ds.add((cat_vg_uri, RDF.type, OLIS.VirtualGraph, SystemGraphURI))
        cat_ds.add((cat_rg_uri, RDF.type, OLIS.RealGraph, SystemGraphURI))
        cat_ds.add((cat_vg_uri, SCHEMA.name, Literal(f"VirtualGraph for Catalogue {catalog_def['token']}"), SystemGraphURI))
        cat_ds.add((cat_vg_uri, OLIS.isAliasFor, cat_rg_uri, SystemGraphURI))
        cat_ds.add((ALL_CATALOGUES_VG, OLIS.isAliasFor, cat_vg_uri, SystemGraphURI))
        for content_graph_detail in cat_details.content_graphs:
            content_rg_uri: Optional[URIRef] = content_graph_detail.graph_name
            if content_rg_uri is None:
                if VOCABS_IN_INDIVIDUAL_RG:
                    content_rg_uri = URIRef(str(content_graph_detail.vocab_uri).rstrip("/#"))
                else:
                    content_rg_uri = cat_rg_uri
            if content_rg_uri == cat_vg_uri:
                raise RuntimeError("Vocab/Content real-graph URI cannot be the same as the catalogue virtual-graph URI.")
            for (s, p, o) in content_graph_detail.graph:
                cat_ds.add((s, p, o, content_rg_uri))
            cat_ds.add((ALL_VOCABS_VG, OLIS.isAliasFor, content_rg_uri, SystemGraphURI))
            if content_rg_uri != cat_rg_uri:
                cat_ds.add((content_rg_uri, RDF.type, OLIS.RealGraph, SystemGraphURI))
                cat_ds.add((cat_vg_uri, OLIS.isAliasFor, content_rg_uri, SystemGraphURI))

        with open(f"./generated/{catalog_def['token']}_all.nq", "wb") as f:
            cat_ds.serialize(f, format="nquads")


async def fix_self_same_as(catalogue_tokens: Optional[list[str]] = None, apply: bool = False) -> int:
    catalog_defs = config.get_value("catalogues", None)
    if catalog_defs is None or len(catalog_defs) == 0:
        raise Exception("No catalogues defined")

    token_filter = set(catalogue_tokens) if catalogue_tokens else None
    target_catalogues = list(iter_graphdb_catalogues(catalog_defs, tokens=token_filter))
    if len(target_catalogues) == 0:
        if token_filter:
            print("No matching GraphDB SPARQL catalogues found: " + ", ".join(sorted(token_filter)))
        else:
            print("No GraphDB SPARQL catalogues found.")
        return 0

    total_applied_subjects = 0
    for catalog_def in target_catalogues:
        result = await cleanup_catalogue_self_same_as(catalog_def, apply=apply)
        count = len(result.subjects)
        total_applied_subjects += count if result.applied else 0
        mode = "APPLIED" if result.applied else "DRY RUN"
        print(f"[{mode}] {result.token}: found {count} self-referential owl:sameAs triples")
        print(f"  query endpoint: {result.query_endpoint}")
        print(f"  update endpoint: {result.update_endpoint}")
        for subject in result.subjects:
            print(f"  <{subject}> owl:sameAs <{subject}>")

    if apply:
        print(f"Applied delete updates for {total_applied_subjects} matching subjects.")
    else:
        print("Dry run only. Re-run with --apply to delete these triples upstream.")
    return 0


def entrypoint(argv: Optional[list[str]] = None) -> int:
    import argparse
    import asyncio
    parser = argparse.ArgumentParser(description="Build BDR reference-data outputs and run maintenance tasks.")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Build catalogue and vocabulary output files.")
    build_parser.add_argument(
        "--catalogue",
        action="append",
        dest="catalogues",
        help="Catalogue token to build. May be supplied more than once. Defaults to every catalogue.",
    )

    fix_parser = subparsers.add_parser(
        "fix-self-sameas",
        help="Remove explicit owl:sameAs triples where subject and object are identical from GraphDB sources.",
    )
    fix_parser.add_argument(
        "--catalogue",
        action="append",
        dest="catalogues",
        help="Catalogue token to fix. May be supplied more than once. Defaults to every GraphDB SPARQL catalogue.",
    )
    fix_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the GraphDB DELETE updates. Without this, the command only reports matching triples.",
    )

    args = parser.parse_args(argv)
    loop = asyncio.get_event_loop()
    try:
        if args.command == "fix-self-sameas":
            return loop.run_until_complete(fix_self_same_as(catalogue_tokens=args.catalogues, apply=args.apply))
        loop.run_until_complete(build_catalogues(catalogue_tokens=getattr(args, "catalogues", None)))
    except Exception as e:
        import traceback
        traceback.print_tb(e.__traceback__)
        print(e)
        return 1
    return 0
