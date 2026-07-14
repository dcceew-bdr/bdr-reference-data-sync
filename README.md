# BDR Reference Data Sync repo

This repo contains the code for the BDR Reference Data Sync scripts.

This set of scripts harvests and catalogs Controlled Vocabularies, Instance Lists, Codelists, Ontologies and other reference data for the BDR from a variety of sources.

## Maintenance commands

Audit explicit `owl:sameAs` triples where the subject and object are identical in configured GraphDB sources:

```shell
python main.py fix-self-sameas
```

Apply the upstream delete through the configured GraphDB update endpoint:

```shell
python main.py fix-self-sameas --catalogue tern-cv --apply
```
