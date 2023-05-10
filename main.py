from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, \
    TextClassificationPipeline
from transformers import pipeline

from custom_model import BertClassifier

from transformers import Pipeline


class MyPipeline(TextClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# Press the green button in the gutter to run the script.
def main():
    tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
    model = AutoModelForMaskedLM.from_pretrained("asafaya/bert-base-arabic")

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
    model = AutoModelForSequenceClassification.from_pretrained("asafaya/bert-base-arabic")
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    res = pipe("باريس [MASK] فرنسا.")
    print(res)
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

    from transformers.pipelines import PIPELINE_REGISTRY

    PIPELINE_REGISTRY.register_pipeline(
        "new-task",
        pipeline_class=MyPipeline,
        pt_model=custom_model,
    )

    # TODO: consider the pipeline of text classfication

    pipe = pipeline("new-task", model=custom_model, tokenizer=tokenizer)
    res = pipe(["without mask", "باريس [MASK] فرنسا."])
    print(res)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
