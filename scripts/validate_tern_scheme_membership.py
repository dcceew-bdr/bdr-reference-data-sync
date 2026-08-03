#!/usr/bin/env python3
import sys

import rdflib
from rdflib.namespace import RDF, SKOS


GRAPH_URI = rdflib.URIRef(
    "https://linked.data.gov.au/dataset/bdr/catalogues/tern-cv-rg"
)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} GENERATED_NQUADS", file=sys.stderr)
        return 2

    dataset = rdflib.Dataset()
    dataset.parse(sys.argv[1], format="nquads")
    graph = dataset.graph(GRAPH_URI)
    findings: list[tuple[str, rdflib.term.Identifier, int]] = []

    for concept in set(graph.subjects(RDF.type, SKOS.Concept)):
        count = len(set(graph.objects(concept, SKOS.inScheme)))
        if count > 1:
            findings.append(("multiple concept schemes", concept, count))
        elif count == 0:
            findings.append(("missing concept scheme", concept, count))
    for collection in set(graph.subjects(RDF.type, SKOS.Collection)):
        count = len(set(graph.objects(collection, SKOS.inScheme)))
        if count == 0:
            findings.append(("missing collection scheme", collection, count))

    print("issue\tresource\tschemeCount")
    for issue, resource, count in sorted(findings, key=lambda row: (row[0], str(row[1]))):
        print(f"{issue}\t{resource}\t{count}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
