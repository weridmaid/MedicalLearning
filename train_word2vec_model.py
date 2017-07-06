#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# import yaml

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    # if len(sys.argv) < 4:
    #     print globals()['__doc__'] % locals()
    #     sys.exit(1)
    # inp, outp1, outp2 = sys.argv[1:4]
    inp = 'all_data_for_w2v.txt'
    outp1 = 'w2v_50d_min10.model'
    outp2 = 'w2v_50d_min10.model.vector'
    # config = yaml.load(open("data_process_config.yaml"))  # 加载配置文件

    model = Word2Vec(LineSentence(inp), size=50, window=2, min_count=10,
            workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    model.save(outp1)
    model.save_word2vec_format(outp2, binary=False)
