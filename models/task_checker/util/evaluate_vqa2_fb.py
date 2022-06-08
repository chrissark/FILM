import re
import pymorphy2
morph = pymorphy2.MorphAnalyzer()


class VQAEval:
    def __init__(self, vqa, vqaRes, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa = vqa
        self.vqaRes = vqaRes
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                             "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                             "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                             "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                             "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                             "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                             "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                             "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                             "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                             "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                             "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                             "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                             "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                             "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                             "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manualMap = {'none': '0',
                              'zero': '0',
                              'one': '1',
                              'two': '2',
                              'three': '3',
                              'four': '4',
                              'five': '5',
                              'six': '6',
                              'seven': '7',
                              'eight': '8',
                              'nine': '9',
                              'ten': '10'
                            }
        self.manualMap_ru = {'ноль': '0',
                             'нисколько': '0',
                             'никакой': '0',
                             'один': '1',
                             'два': '2',
                             'три': '3',
                             'четыре': '4',
                             'пять': '5',
                             'шесть': '6',
                             'семь': '7',
                             'восемь': '8',
                             'девять': '9',
                             'десять': '10'
                            }
        self.articles = ['a', 'an', 'the']
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']

    def evaluate(self, quesIds, true_json, pred_json):

        true_res = true_json
        pred_res = pred_json

        accQA = []

        for quesId in quesIds:

            lang = true_res[quesId]['lang']
            resAns = pred_res[quesId]
            resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            true_acc = []
            true_ans = true_res[quesId]['answer']

            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns, lang)

            matchingAns = []
            if resAns == true_ans:
                matchingAns.append(resAns)

            acc = min(1, float(len(matchingAns)))
            true_acc.append(acc)

            avgGTAcc = float(sum(true_acc))/len(true_acc)
            accQA.append(avgGTAcc)

            self.setEvalQA(quesId, avgGTAcc)

        self.setAccuracy(accQA)


    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText


    def processDigitArticle(self, inText, lang):
        outText = []
        tempText = inText.lower().split()

        if lang == 'ru':
            for word in tempText:
                word_parsed = morph.parse(word)[0]
                lemma = word_parsed.normal_form
                lemma = self.manualMap_ru.setdefault(lemma, lemma)
                outText.append(lemma)
        else:
            for word in tempText:
                word = self.manualMap.setdefault(word, word)
                if word not in self.articles:
                    outText.append(word)
                else:
                    pass
            for wordId, word in enumerate(outText):
                if word in self.contractions:
                    outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText


    def setAccuracy(self, accQA):
        self.accuracy['overall'] = round(100*float(sum(accQA))/len(accQA), self.n)


    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100*acc, self.n)


# def vqa_evaluate(vqa_result):
#     true_json, pred_json = {}, {}
#     for i, row in vqa_result.iterrows():
#         true_json[str(i)] = {
#             "answer": row['gt_output'],
#             "lang": "en"
#         }
#         pred_json[str(i)] = row['pred_output']
#     quesIds = list(true_json.keys())
#
#     assert quesIds == list(pred_json.keys()), 'The order of predictions doesn’t match the order of targets!'
#
#     vqaEval = VQAEval(true_json, pred_json, n=2)
#     vqaEval.evaluate(quesIds, true_json, pred_json)
#
#     return vqaEval.accuracy['overall']

def vqa_evaluate(true_json, pred_json, pred_json_has_lang_key=False):
    quesIds = list(true_json.keys())
    assert quesIds == list(pred_json.keys()), 'The order of predictions doesn’t match the order of targets!'

    if pred_json_has_lang_key:
        pred_json = {key: value['answer'] for key, value in pred_json.items()}

    vqaEval = VQAEval(true_json, pred_json, n=2)
    vqaEval.evaluate(quesIds, true_json, pred_json)

    return vqaEval.accuracy['overall']