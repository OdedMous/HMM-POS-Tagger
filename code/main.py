from nltk.corpus import brown
import pandas as pd

from code.word_counting import *
from code.models import Baseline_Model
from code.models import HMM_Model
from code.pseudowords import replace_to_pseudowords

def fix_tag(multi_tag):
    """
    Get the first tag from a multiple tags.
    """
    temp = multi_tag.split('-')[0]
    return temp.split('+')[0]

def get_data(p=80):
    """
    Get Brown news text, and divides it into p% training set and (1-p)% test set.
    :param p: percentage of training set.
    """

    brown_news_text = [sentence for sentence in brown.tagged_sents(categories='news')]

    # clean multiple tags
    for sentence in brown_news_text:
        for i in range(len(sentence)):
            sentence[i] = (sentence[i][WORD], fix_tag(sentence[i][TAG]))

    # divide data
    size = int(len(brown_news_text) * p/100)
    training = brown_news_text[:size]
    test = brown_news_text[size:]

    return [brown_news_text, training, test]

def save_errors():
    """
    Saves the error results in Excel file in the current directory.
    In addition, a visual chart of the errors will be created and will be saved there too.
    """
    known = {'baseline': x1, 'HMM': x2, 'smoothing': x3, 'pseudowords': x4, 'smoothing&pseudowords': x5}
    unknown = {'baseline': y1, 'HMM': y2, 'smoothing': y3, 'pseudowords': y4, 'smoothing&pseudowords': y5}
    total = {'baseline': z1, 'HMM': z2, 'smoothing': z3, 'pseudowords': z4, 'smoothing&pseudowords': z5}

    data = [known, unknown, total]
    index = ['known', 'unknown', 'total']

    # Create a Pandas dataframe from the data.
    df = pd.DataFrame(data, index=index)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    sheet_name = 'Sheet1'
    writer = pd.ExcelWriter('pandas_chart_columns.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name=sheet_name)

    # Access the XlsxWriter workbook and worksheet objects from the dataframe.
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    # Create a chart object.
    chart = workbook.add_chart({'type': 'column'})

    # Some alternative colors for the chart.
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']

    # Configure the series of the chart from the dataframe data.
    for col_num in range(1, len(known) + 1):
        chart.add_series({
            'name': ['Sheet1', 0, col_num],
            'categories': ['Sheet1', 1, 0, 4, 0],
            'values': ['Sheet1', 1, col_num, 4, col_num],
            'fill': {'color': colors[col_num - 1]},
            'overlap': -10,
        })

    # Configure the chart axes.
    chart.set_x_axis({'name': 'Error Type'})
    chart.set_y_axis({'name': 'Error Rate', 'major_gridlines': {'visible': False}})

    # Insert the chart into the worksheet.
    worksheet.insert_chart('H2', chart)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


if __name__ == "__main__":

    # Extract data
    news_text, training_set, test_set = get_data(p=80)
    flat_training_set = [item for sublist in training_set for item in sublist]
    flat_test_set = [item for sublist in test_set for item in sublist]

    counts = Counts(flat_training_set, flat_test_set)

    print("0/5 ")
    # Errors for MLE tag baseline
    baseline_model = Baseline_Model(counts)
    x1, y1, z1 = baseline_model.errors(flat_test_set, news_text)
    print("1/5 Done")

    # Errors for bigram HMM tagger
    hmm_model = HMM_Model(counts)
    x2, y2, z2 = hmm_model.errors(flat_training_set, flat_test_set)
    print("2/5 Done")

    # Errors for bigram HMM tagger + add1 smoothing
    x3, y3, z3 = hmm_model.errors(flat_training_set, flat_test_set, add_one=True)
    print("3/5 Done")

    # Errors for bigram HMM tagger + using pseudowords
    new_training, new_test = replace_to_pseudowords(flat_training_set, flat_test_set, counts)
    counts = Counts(new_training, new_test)
    hmm_model = HMM_Model(counts)
    x4, y4, z4 = hmm_model.errors(new_training, new_test)
    print("4/5 Done")

    # Errors for bigram HMM tagger + using pseudowords + add1 smoothing
    x5, y5, z5 = hmm_model.errors(new_training, new_test, add_one=True)
    print("5/5 Done")

    # Save errors in Excel file
    save_errors()

