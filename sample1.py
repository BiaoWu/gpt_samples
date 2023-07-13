from transformers import AutoTokenizer, AutoModel


def main():
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
    model = AutoModel.from_pretrained("lmsys/vicuna-7b-v1.3")
    inputs = tokenizer("Hello world!", return_tensors="pt")
    outputs = model(**inputs)
    print("outputs : %s" % outputs)


if __name__ == '__main__':
    main()
