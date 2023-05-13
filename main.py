from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, \
    TextClassificationPipeline, AutoConfig

from transformers import pipeline

from transformers import pipeline

from custom_model import CustomModel
from transformers.pipelines import PIPELINE_REGISTRY

from custom_models import BertClassifier


class MyModel(AutoModelForSequenceClassification):
    pass


class MyPipeline(TextClassificationPipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, preprocess_kwargs

    def preprocess(self, inputs, **tokenizer_kwargs):
        tokenizer_kwargs.pop('maybe_arg')
        return super().preprocess(inputs, **tokenizer_kwargs)

    def forward(self, model_inputs, **forward_params):
        return super().forward(model_inputs, **forward_params)

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True, maybe_arg=5):
        return super().postprocess(model_outputs, function_to_apply, top_k, _legacy)

def CustomTextClassificationPipeline(TextClassificationPipeline):
    pass



def main():
    bert_classifier = BertClassifier(AutoConfig.from_pretrained('bert-base-uncased'))
    bert_classifier.save_pretrained('./bert_classifier')
    # print(bert_classifier)

    loaded_bert_classifier = BertClassifier.from_pretrained('./bert_classifier')
    # print(loaded_bert_classifier)


    PIPELINE_REGISTRY.register_pipeline(
        "custom-text-classification",
        pipeline_class=TextClassificationPipeline,
        pt_model=BertClassifier,
    )

    orig_pipeline =  pipeline(task='text-classification', model='bert-base-uncased', tokenizer='bert-base-uncased')
    print(orig_pipeline("text"))

    custom_pipeline =  pipeline(task='custom-text-classification', model=loaded_bert_classifier, tokenizer='bert-base-uncased')
    print(custom_pipeline("text"))



    return

    custom_pipeline = pipeline(task='custom-text-classification', model=loaded_bert_classifier, tokenizer='bert-base-uncased')
    res = custom_pipeline('text')
    return

    # tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
    # custom_model = MyModel.from_pretrained("asafaya/bert-base-arabic")
    # custom_model = CustomModel(2)
    #
    # PIPELINE_REGISTRY.register_pipeline(
    #     "new-task",
    #     pipeline_class=MyPipeline,
    #     pt_model=custom_model,
    # )
    # pipe = pipeline("new-task", model=custom_model, tokenizer=tokenizer, maybe_arg=5)
    # res = pipe("My sentence")
    # print(res)
    # return

    # model = AutoModelForMaskedLM.from_pretrained("asafaya/bert-base-arabic")

    # pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    # res = pipe("text")
    # print(res)
    # # error
    # # The model 'BertForMaskedLM' is not supported for text-classification. Supported models are ['AlbertForSequenceClassification', 'BartForSequenceClassification', 'BertForSequenceClassification', 'BigBirdForSequenceClassification', 'BigBirdPegasusForSequenceClassification', 'BloomForSequenceClassification', 'CamembertForSequenceClassification', 'CanineForSequenceClassification', 'ConvBertForSequenceClassification', 'CTRLForSequenceClassification', 'Data2VecTextForSequenceClassification', 'DebertaForSequenceClassification', 'DebertaV2ForSequenceClassification', 'DistilBertForSequenceClassification', 'ElectraForSequenceClassification', 'ErnieForSequenceClassification', 'ErnieMForSequenceClassification', 'EsmForSequenceClassification', 'FlaubertForSequenceClassification', 'FNetForSequenceClassification', 'FunnelForSequenceClassification', 'GPT2ForSequenceClassification', 'GPT2ForSequenceClassification', 'GPTBigCodeForSequenceClassification', 'GPTNeoForSequenceClassification', 'GPTNeoXForSequenceClassification', 'GPTJForSequenceClassification', 'IBertForSequenceClassification', 'LayoutLMForSequenceClassification', 'LayoutLMv2ForSequenceClassification', 'LayoutLMv3ForSequenceClassification', 'LEDForSequenceClassification', 'LiltForSequenceClassification', 'LlamaForSequenceClassification', 'LongformerForSequenceClassification', 'LukeForSequenceClassification', 'MarkupLMForSequenceClassification', 'MBartForSequenceClassification', 'MegaForSequenceClassification', 'MegatronBertForSequenceClassification', 'MobileBertForSequenceClassification', 'MPNetForSequenceClassification', 'MvpForSequenceClassification', 'NezhaForSequenceClassification', 'NystromformerForSequenceClassification', 'OpenAIGPTForSequenceClassification', 'OPTForSequenceClassification', 'PerceiverForSequenceClassification', 'PLBartForSequenceClassification', 'QDQBertForSequenceClassification', 'ReformerForSequenceClassification', 'RemBertForSequenceClassification', 'RobertaForSequenceClassification', 'RobertaPreLayerNormForSequenceClassification', 'RoCBertForSequenceClassification', 'RoFormerForSequenceClassification', 'SqueezeBertForSequenceClassification', 'TapasForSequenceClassification', 'TransfoXLForSequenceClassification', 'XLMForSequenceClassification', 'XLMRobertaForSequenceClassification', 'XLMRobertaXLForSequenceClassification', 'XLNetForSequenceClassification', 'XmodForSequenceClassification', 'YosoForSequenceClassification'].
    #
    # pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    # res = pipe("باريس [MASK] فرنسا.")
    # print(res)
    #
    # model = AutoModelForSequenceClassification.from_pretrained("asafaya/bert-base-arabic")
    # pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    # res = pipe("باريس [MASK] فرنسا.")
    # print(res)
    # res = pipe("without mask")
    # print(res)
    #
    # res = pipe(["without mask", "باريس [MASK] فرنسا."])
    # print(res)

    # Trying our own model
    custom_model = BertClassifier(num_labels=20)
    # pipe = pipeline("text-classification", model=custom_model, tokenizer=tokenizer)
    # The model 'BertClassifier' is not supported for text-classification. Supported models are ['AlbertForSequenceClassification', 'BartForSequenceClassification', 'BertForSequenceClassification', 'BigBirdForSequenceClassification', 'BigBirdPegasusForSequenceClassification', 'BloomForSequenceClassification', 'CamembertForSequenceClassification', 'CanineForSequenceClassification', 'ConvBertForSequenceClassification', 'CTRLForSequenceClassification', 'Data2VecTextForSequenceClassification', 'DebertaForSequenceClassification', 'DebertaV2ForSequenceClassification', 'DistilBertForSequenceClassification', 'ElectraForSequenceClassification', 'ErnieForSequenceClassification', 'ErnieMForSequenceClassification', 'EsmForSequenceClassification', 'FlaubertForSequenceClassification', 'FNetForSequenceClassification', 'FunnelForSequenceClassification', 'GPT2ForSequenceClassification', 'GPT2ForSequenceClassification', 'GPTBigCodeForSequenceClassification', 'GPTNeoForSequenceClassification', 'GPTNeoXForSequenceClassification', 'GPTJForSequenceClassification', 'IBertForSequenceClassification', 'LayoutLMForSequenceClassification', 'LayoutLMv2ForSequenceClassification', 'LayoutLMv3ForSequenceClassification', 'LEDForSequenceClassification', 'LiltForSequenceClassification', 'LlamaForSequenceClassification', 'LongformerForSequenceClassification', 'LukeForSequenceClassification', 'MarkupLMForSequenceClassification', 'MBartForSequenceClassification', 'MegaForSequenceClassification', 'MegatronBertForSequenceClassification', 'MobileBertForSequenceClassification', 'MPNetForSequenceClassification', 'MvpForSequenceClassification', 'NezhaForSequenceClassification', 'NystromformerForSequenceClassification', 'OpenAIGPTForSequenceClassification', 'OPTForSequenceClassification', 'PerceiverForSequenceClassification', 'PLBartForSequenceClassification', 'QDQBertForSequenceClassification', 'ReformerForSequenceClassification', 'RemBertForSequenceClassification', 'RobertaForSequenceClassification', 'RobertaPreLayerNormForSequenceClassification', 'RoCBertForSequenceClassification', 'RoFormerForSequenceClassification', 'SqueezeBertForSequenceClassification', 'TapasForSequenceClassification', 'TransfoXLForSequenceClassification', 'XLMForSequenceClassification', 'XLMRobertaForSequenceClassification', 'XLMRobertaXLForSequenceClassification', 'XLNetForSequenceClassification', 'XmodForSequenceClassification', 'YosoForSequenceClassification'].
    # model = AutoModelForSequenceClassification.from_pretrained("asafaya/bert-base-arabic")

    my_model = MyModel.from_pretrained("asafaya/bert-base-arabic")

    # from transformers.pipelines import PIPELINE_REGISTRY

    # TODO: consider the pipeline of text classfication

    pipe = pipeline("new-task", model=my_model, tokenizer=tokenizer)
    res = pipe(["without mask", "باريس [MASK] فرنسا."])
    print(res)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
