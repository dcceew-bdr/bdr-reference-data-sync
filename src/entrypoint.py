

if __name__ == '__main__':
    import sys
    print("Run with python main.py or python -m src")
    sys.exit(1)

from . import config
from .catalog import build_catalog
from .voc_graph import make_voc_graph

async def build_catalogs():
    catalog_defs = config.get_value("catalogs", None)
    if catalog_defs is None or len(catalog_defs) == 0:
        raise Exception("No catalogs defined")

    for catalog_def in catalog_defs:
        if 'token' not in catalog_def:
            raise RuntimeError("Catalog entry does not have name property.")
        print(f"building {catalog_def['token']}")
        cat_details = await build_catalog(catalog_def)
        cat_ds = make_voc_graph(multigraph=True)
        cat_uri = cat_details.cat_uri
        for (s, p, o) in cat_details.graph:
            cat_ds.add((s, p, o, cat_uri))
        for content_graph in cat_details.content_graphs:
            for (s, p, o) in content_graph.graph:
                cat_ds.add((s, p, o, cat_uri))
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


