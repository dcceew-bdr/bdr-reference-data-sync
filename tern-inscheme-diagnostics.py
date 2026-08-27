from pathlib import Path
import rdflib
from rdflib.namespace import RDF, SKOS

root = Path("generated/tern-cv/vocabularies")

all_graph = rdflib.Graph()
per_file = {}

print("Loading TTL files...")
for path in sorted(root.glob("*.ttl")):
    g = rdflib.Graph()
    try:
        g.parse(path, format="turtle")
    except Exception as e:
        print(f"ERROR parsing {path}: {e}")
        continue

    per_file[path] = g
    for t in g:
        all_graph.add(t)

print("\n1. Files containing more than one ConceptScheme")
for path, g in per_file.items():
    schemes = set(g.subjects(RDF.type, SKOS.ConceptScheme))
    if len(schemes) > 1:
        print(f"{path}: {len(schemes)} ConceptSchemes")
        for s in sorted(str(x) for x in schemes):
            print(f"  {s}")

print("\n2. Concepts with skos:inScheme pointing to ConceptSchemes not defined in the same file")
for path, g in per_file.items():
    schemes_in_file = set(g.subjects(RDF.type, SKOS.ConceptScheme))
    for c in set(g.subjects(RDF.type, SKOS.Concept)):
        for scheme in g.objects(c, SKOS.inScheme):
            if scheme not in schemes_in_file:
                print(f"{path}: {c} inScheme {scheme}")

print("\n3. Concepts with multiple skos:inScheme memberships in the same file")
for path, g in per_file.items():
    for c in set(g.subjects(RDF.type, SKOS.Concept)):
        schemes = set(g.objects(c, SKOS.inScheme))
        if len(schemes) > 1:
            print(f"{path}: {c}")
            for s in sorted(str(x) for x in schemes):
                print(f"  {s}")

print("\n4a. Concepts with no skos:inScheme in their own file")
for path, g in per_file.items():
    for c in set(g.subjects(RDF.type, SKOS.Concept)):
        schemes = set(g.objects(c, SKOS.inScheme))
        if len(schemes) == 0:
            print(f"{path}: {c}")

print("\n4b. Concepts with no skos:inScheme across all generated vocab docs")
for c in set(all_graph.subjects(RDF.type, SKOS.Concept)):
    schemes = set(all_graph.objects(c, SKOS.inScheme))
    if len(schemes) == 0:
        print(c)

print("\nDone.")
