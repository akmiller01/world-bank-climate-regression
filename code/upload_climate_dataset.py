from huggingface_hub import login
from datasets import Dataset
from dotenv import load_dotenv
import os
from collections import Counter
import pandas as pd


def bin_labels(example):
    example['class_label'] = example['Climate change'] == 0.
    return example


def main():
    df = pd.read_csv("./input/wb_api_climate_percentages.csv")
    starting_rows = df.shape[0]
    print("Starting rows:", starting_rows)
    # Limit to post 2011 for climate accuracy
    df = df[(df['fiscalyear'] >= 2011) | (df['fiscalyear'] == 0)]
    print("Rows removed by year filter:", starting_rows - df.shape[0])
    df['text'] = (df['project_name'] + ' ' + df['pdo'] + ' ' + df['project_abstract']).str.strip()
    # Remove obvious false negatives
    # Keywords to search for in the 'text' column
    keywords = ["climate change", "adaptation", "mitigation", "renewable", "natural disaster", "disaster risk"]

    # Create a filter
    negative_mask = (df['Climate change'] == 0) & df['text'].str.contains('|'.join(keywords), case=False)

    # Apply the filter to the DataFrame
    prefilter_rows = df.shape[0]
    df = df[~negative_mask]
    postfilter_rows = df.shape[0]
    print("Rows removed by obvious false negative filter:", prefilter_rows - postfilter_rows)
    # De-duplicate
    prededup_rows = df.shape[0]
    df = df.drop_duplicates(subset=['text'])
    postdedup_rows = df.shape[0]
    print("Rows removed by de-duplication:", prededup_rows - postdedup_rows)
    print("Ending rows:", df.shape[0])
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.map(bin_labels)

    # Remove blanks
    dataset = dataset.filter(lambda example: example['text'] is not None)

    count = Counter()
    count.update(dataset['class_label'])
    print(count)

    dataset = dataset.class_encode_column('class_label').train_test_split(
        test_size=0.2,
        stratify_by_column="class_label",
        shuffle=True,
        seed=1337
    )
    dataset = dataset.remove_columns(['class_label', 'proj_id', 'project_name', 'pdo', 'project_abstract'])
    dataset = dataset.rename_columns({'Adaptation': 'Climate adaptation', 'Mitigation': 'Climate mitigation'})
    dataset.push_to_hub("alex-miller/wb-climate-percentage")


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()
