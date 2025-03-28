
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
from .voc_graph import make_voc_graph

# Customisable constants
ALL_VOCABS = URIRef("urn:bdr:all-vocabularies")
ALL_CATALOGUES = URIRef("urn:bdr:all-catalogues")
VOCABS_IN_INDIVIDUAL_RG = False

# Non-customisable constants
OLIS = Namespace("https://olis.dev/")
SystemGraphURI = OLIS.system



async def build_catalogs():
    catalog_defs = config.get_value("catalogs", None)
    if catalog_defs is None or len(catalog_defs) == 0:
        raise Exception("No catalogs defined")

    for catalog_def in catalog_defs:
        if 'token' not in catalog_def:
            raise RuntimeError("Catalog entry does not have token property.")
        print(f"Building Catalogue: {catalog_def['token']}", flush=True)
        cat_details = await build_catalog(catalog_def)
        cat_ds = make_voc_graph(multigraph=True)
        cat_vg_uri: URIRef = cat_details.cat_uri
        cat_rg_uri = cat_details.graph_name
        if cat_rg_uri is None:
            # Fall-back to auto-generated, because this cannot be the same as the catalog name
            cat_rg_uri = URIRef(str(cat_vg_uri).rstrip("/#") + "-catalogue")
        if cat_vg_uri == cat_rg_uri:
            raise RuntimeError(f"Catalog real-graph and virtual-graph URIs cannot be the same.\n<{cat_vg_uri}>==<{cat_rg_uri}>")
        for (s, p, o) in cat_details.graph:
            cat_ds.add((s, p, o, cat_rg_uri))
        cat_ds.add((ALL_VOCABS, RDF.type, OLIS.VirtualGraph, SystemGraphURI))
        cat_ds.add((ALL_CATALOGUES, RDF.type, OLIS.VirtualGraph, SystemGraphURI))
        cat_ds.add((cat_vg_uri, RDF.type, OLIS.VirtualGraph, SystemGraphURI))
        cat_ds.add((cat_rg_uri, RDF.type, OLIS.RealGraph, SystemGraphURI))
        cat_ds.add((cat_vg_uri, SCHEMA.name, Literal(f"VirtualGraph for Catalogue {catalog_def['token']}"), SystemGraphURI))
        cat_ds.add((cat_vg_uri, OLIS.isAliasFor, cat_rg_uri, SystemGraphURI))
        cat_ds.add((ALL_CATALOGUES, OLIS.isAliasFor, cat_vg_uri, SystemGraphURI))
        for vocab_graph_detail in cat_details.content_graphs:
            vocab_rg_uri: Optional[URIRef] = vocab_graph_detail.graph_name
            if vocab_rg_uri is None:
                if VOCABS_IN_INDIVIDUAL_RG:
                    vocab_rg_uri = URIRef(str(vocab_graph_detail.vocab_uri).rstrip("/#"))
                else:
                    vocab_rg_uri = cat_rg_uri
            if vocab_rg_uri == cat_vg_uri:
                raise RuntimeError("Vocab real-graph URI cannot be the same as the catalog virtual-graph URI.")
            for (s, p, o) in vocab_graph_detail.graph:
                cat_ds.add((s, p, o, vocab_rg_uri))
            cat_ds.add((ALL_VOCABS, OLIS.isAliasFor, vocab_rg_uri, SystemGraphURI))
            if vocab_rg_uri != cat_rg_uri:
                cat_ds.add((vocab_rg_uri, RDF.type, OLIS.RealGraph, SystemGraphURI))
                cat_ds.add((cat_vg_uri, OLIS.isAliasFor, vocab_rg_uri, SystemGraphURI))

        with open(f"./generated/{catalog_def['token']}_all.nq", "wb") as f:
            cat_ds.serialize(f, format="nquads")


def entrypoint() -> int:
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(build_catalogs())
    except Exception as e:
        import traceback
        traceback.print_tb(e.__traceback__)
        print(e)
        return 1
    return 0


