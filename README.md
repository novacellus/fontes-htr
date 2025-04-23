# eFontes Project
## HTR Models
Repository contains the HTR pipeline of the eFontes project (efontes.pl). Two types of data are being used to train models:
* incunabula and early prints ([impr](impr) folder)
* manuscripts
  * [mss](mss) folder contains transcription prepared by the eFontes team under supervision of Iwona Krawczyk
  * [mss_transkribathon](mss_transkribathon) folder contains transcription prepared by the participants of the 1st eFontes Transkribathon organized by Iwona Krawczyk and Jagoda Marszałek.

The output [impr/output](impr/output) and [mss/output](mss/output) folders provide [kraken](https://github.com/mittagessen/kraken)-based pipeline: conversion scripts (PAGE XML v. 2009 > 2019 > ALTO XML), Jupyter notebooks used for dataset inspection and evaluation, as well as binary datasets, XML source data (before and after repolygonization), and models built thereof.
Suffixes *abbreviated and *expanded refer to 2 modes in which data were annotated: with abbreviations expanded and not.
