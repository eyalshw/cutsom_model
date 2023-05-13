from transformers import pipeline


def main():
    unmasker = pipeline('fill-mask', model='bert-base-uncased')
    print(unmasker("The man worked as a [MASK]."))


if __name__ == '__main__':
    main()
