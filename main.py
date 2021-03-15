from core import BOCW

if __name__ == '__main__':
    extractor = BOCW(file_path='datasets/xeno_dataset_base.csv',
                     output_folder="results",
                     text_column="tweet",
                     class_column="class")

    extractor.extract(grams=(1, 2, 3),
                      majority_class="no_xeno",
                      keep=True,
                      unique_f=True,
                      max_words=6)
