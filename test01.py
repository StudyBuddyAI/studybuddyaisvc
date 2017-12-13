import argparse
from allennlp.service.predictors import DemoModel


print("Start.... test")

# Machine Comprehension (MC) models answer natural language questions by selecting an answer span within an evidence text.
# The AllenNLP MC model is a reimplementation of BiDAF (Seo et al, 2017), or Bi-Directional Attention Flow
# , a widely used MC baseline that achieves near state-of-the-art accuracies on the SQuAD dataset.
#bidaf_model = DemoModel('../../allennlp/bidaf-model-2017.09.15-charpad.tar.gz','machine-comprehension')
#bidaf_model = DemoModel('../../allennlp/train_out/model01.tar.gz','machine-comprehension')
#bidaf_model = DemoModel('../../allennlp/train_out/model02.tar.gz','machine-comprehension')
bidaf_model = DemoModel('../../allennlp/train_out/model04.tar.gz','machine-comprehension')

# predictor
predictor = bidaf_model.predictor()

# Example 1
data = {
    "passage": "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight."
    ,"question": "How many partially reusable launch systems were developed?"
}
prediction = predictor.predict_json(data)
print(prediction)
print(prediction['best_span_str'])


# Example 2
data = {
    "passage" : "Data science, also known as data-driven science, is an interdisciplinary field about scientific methods, processes, and systems to extract knowledge or insights from data in various forms, either structured or unstructured, similar to data mining. Machine learning is a field of computer science that gives computers the ability to learn without being explicitly programmed. Arthur Samuel, an American pioneer in the field of computer gaming and artificial intelligence, coined the term Machine Learning in 1959 while at IBM. Statistical learning refers to a vast set of tools for understanding data."
    ,"question": "what is machine learning?"
}
prediction = predictor.predict_json(data)
print(prediction)
print(prediction['best_span_str'])


print("End.... test")