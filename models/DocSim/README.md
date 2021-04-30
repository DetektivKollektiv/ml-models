# Machine Learning Model for predicting similarities between documents

This model is used for the search of factchecks and for tagging submitted items.

## Training

The training data is a csv file with a column "claim_data" containing strings. 
The name of the file is specified in DocSim-params.json
The training file is located in a bucket specified in inputData.json

### Taxonomy

Beside the model a report for adopting a taxonomy is created.
The uri for the taxonomy is specified in DocSim-params.json
The taxonomy contains "similarity-threshold" which gives the threshold for similarity scores so that an item content is considered similar enough to a tag term to set the correpsonding tag.
The taxonomy contains a list "excluded-terms" to define terms which should be ignored in an item content as they would result in wrong tags.
The remaining objects specify categories of tags, tags and lists of words indicating a tag.
This is an example of a taxonomy:
{
    "similarity-threshold": 0.5,
    "gesundheit": {
        "masken": ["masken", "maske", "ffp2"],
        "sars-cov-2": ["corona", "pandemie", "covid", "coronavirus"],
    },
    "excluded-terms": ["biologen", "astronomen"]
}

If one of these words is found in an item content, than the corresponding tag is set.
For example an item content "Corona ist schlimmer als Grippe" would get the tag "sars-cov-2".
If the similarity score for one of these words is above the threshold, than also the corresponding tag is set.
For example an item content "In diesem Geschäft ist FFP2-Pflicht" would get the tag "masken".

### Taxonomy report

The file name for this report is taxonomy_report.json and is stored in the bucket and path as the taxonomy.
The report lists for terms in the taxonomy the most similar words, which could result in setting a tag.
This is an example for a taxonomy report:
{
    "similarity-threshold": 0.5,
    "masken": {
        "masken": {
            "mundschutz": 0.4607822299003601,
            "alltagsmasken": 0.4599316120147705,
            "schutzmasken": 0.452526718378067,
            "gesichtsmasken": 0.44383952021598816,
            "maskentragen": 0.4317548871040344,
            "alltagsmaske": 0.4269312620162964,
            "mund": 0.4091983735561371,
            "händewaschen": 0.4050191640853882,
            "schutzmaske": 0.37566834688186646
        },
        "maske": {
            "mundschutz": 0.5591799020767212,
            "mund": 0.4887275993824005,
            "schutzmaske": 0.486825168132782,
            "maskentragen": 0.46549126505851746,
            "alltagsmaske": 0.43309491872787476,
            "atemschutzmaske": 0.41641876101493835,
            "gesichtsmasken": 0.3724503815174103,
            "atemschutz": 0.361208438873291,
            "nase": 0.3589116334915161
        },
        "ffp": {
            "schutzkleidung": 0.5145227909088135,
            "handschuhe": 0.49031880497932434,
            "auszustatten": 0.4618639349937439,
            "bedeckungen": 0.42366668581962585,
            "verweigerer": 0.4026312232017517,
            "schutzmasken": 0.3975842297077179,
            "desinfektion": 0.39173072576522827,
            "beschafft": 0.38890692591667175,
            "händewaschen": 0.381102055311203,
            "ambulante": 0.3713994026184082
        }
    }
}
In this example the word "schutzkleidung" has a similarity score for "ffp" above the threshold and therefore the tag "masken" is set. As "schutzkleidung" is not necessarly a ffp mask, this could result in wrong tags. Therefore "schutzkleidung" should included in the list of "excluded-terms" in the taxonomy.
On the other hand all similarity scores of the word "gesichtsmasken" are below the threshold and therefore the tag "masken" would not be set, if "gesichtsmasken" is in the item content. Therefore the word "gesichtsmasken" could be included in the taxonomy as one of the tag terms. The taxonomy would then look like:
{
    "similarity-threshold": 0.5,
    "gesundheit": {
        "masken": ["masken", "maske", "ffp2", "gesichtsmasken"],
        "sars-cov-2": ["corona", "pandemie", "covid", "coronavirus"],
    },
    "excluded-terms": ["biologen", "astronomen", "schutzkleidung"]
}


## Inference

The output for inference is a similarity score between two lists of strings.

