import ntpath
import os
import xml.etree.ElementTree as ET

import gensim


class DocumentsIterable(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.files = os.listdir(self.dirname)

    def __iter__(self):
        i = 0
        print('processed documents: 0')
        for filename in self.files:
            try:
                tree = ET.parse(os.path.join(self.dirname, filename))
                root = tree.getroot()

                title = root.find("Title").text
                if title is None:
                    title = ""

                body = root.find("Body").text
                if body is None:
                    body = ""

                i += 1
                text = title + os.linesep + body
                if i % 5000 == 0:
                    print('processed documents: %s' % i)
                yield gensim.models.doc2vec.TaggedDocument(
                    words=list(gensim.utils.tokenize(text, lowercase=True, deacc=True)), tags=ntpath.basename(filename))
            except:
                print('error parsing file: %s, using naive parsing' % filename)
                with open(os.path.join(self.dirname, filename), 'r') as file_content:
                    content = file_content.read()
                    text = content.replace("<document>", '').replace("</document>", '').replace("<Title>",
                                                                                                '').replace(
                        "</Title>", '').replace("<Body>", '').replace("</Body>", '')

                    yield gensim.models.doc2vec.TaggedDocument(
                        words=list(gensim.utils.tokenize(text, lowercase=True, deacc=True)), tags=ntpath.basename(filename))
