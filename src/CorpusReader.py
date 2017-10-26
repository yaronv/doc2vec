import itertools
import ntpath
import os
import xml.etree.ElementTree as ET

import gensim
from nltk.corpus import stopwords


class CorpusReader(object):
    def __init__(self, dirnames):
        self.docs = []
        self.dirnames = dirnames
        self.files = [os.listdir(dirname) for dirname in dirnames]

        # make sure the files contain their full path
        for i, files in enumerate(self.files):
            self.files[i] = [os.path.join(self.dirnames[i], f) for f in self.files[i]]

        self.files = list(itertools.chain(*self.files))

    def read_corpus(self):
        i = 0
        print('starting to process docs (iterating)')
        for filename in self.files:
            try:
                tree = ET.parse(filename)
                root = tree.getroot()

                title = root.find("Title").text
                if title is None:
                    title = ""

                body = root.find("Body").text
                if body is None:
                    body = ""

                i += 1
                text = title + os.linesep + body
                if i % 100 == 0:
                    print('processed documents: %s' % i)
                words = gensim.utils.lemmatize(text, stopwords=stopwords.words('english'))
                self.docs.append(gensim.models.doc2vec.TaggedDocument(
                    [w.decode('utf-8') for w in words], tags=[ntpath.basename(filename)]))
            except:
                print('error parsing file: %s, using naive parsing' % filename)
                with open(filename, 'r') as file_content:
                    content = file_content.read()
                    text = content.replace("<document>", '').replace("</document>", '').replace("<Title>",
                                                                                                '').replace(
                        "</Title>", '').replace("<Body>", '').replace("</Body>", '')
                    words = gensim.utils.lemmatize(text, stopwords=stopwords.words('english'))
                    self.docs.append(gensim.models.doc2vec.TaggedDocument(
                        [w.decode('utf-8') for w in words], tags=[ntpath.basename(filename)]))
        return self.docs