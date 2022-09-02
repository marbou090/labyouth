import re
from janome.tokenizer import Tokenizer
import numpy as np
from gensim.models.doc2vec import Doc2Vec

from pyknp import KNP
import pyknp
import pandas as pd
import itertools
import copy

import spacy
from pyknp import Juman
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

from gensim.models.doc2vec import Doc2Vec

import timeit
import tqdm

import warnings
warnings.simplefilter('ignore', FutureWarning)

MIN_LEN=20
MAX_LEN=40


class SentenceCompression:
    def __init__(self):
        self.knp = pyknp.KNP(jumanpp=False)
        self.MIN_LEN=20
        self.MAX_LEN=40  

    def return_df(self):
        cols = ['ID', '見出し','親文節']
        df = pd.DataFrame(index=[], columns=cols)
        return df

    def phrase_analysis(self,text):
        self.knp = pyknp.KNP(jumanpp=False)



        cols = ['ID', '見出し','親文節']
        df = pd.DataFrame(index=[], columns=cols)
        result = self.knp.parse(text)

        for bnst in result.bnst_list():
            record = pd.Series([bnst.bnst_id, "".join(mrph.midasi for mrph in bnst.mrph_list()), bnst.parent_id], index=df.columns)
            df = df.append(record, ignore_index=True)
        
        return df
    
    def count_childroot(self,df):
        #その文節を親文節としてる文節の個数を数えて足したい
        df['子文節個数'] = 0
        for index, row in df.iterrows():
            df_id = df.iloc[index]['ID']
            if not len(df[df['親文節']==df_id]) == 0:
                df.iloc[index,3] = len(df[df['親文節']==df_id])#これを親文節としてるのを列挙し代入する
        return df
    
    def find_rootindex(self,df,index,r_df):
        """
        親文節をたどって部分木から根までのdfを返す
        """
        r_df = r_df.append(df[df['ID']==index])
        df_id = df.iloc[index]['親文節']
        if df_id==-1:
            return r_df
        else:
            return self.find_rootindex(df,df_id,r_df)
    
    def count_word(self,df):
        """
        DataFrameをうけとって、'見出し'の文字数をカウントする
        """
        text=''
        for index,row in df.iterrows():
          text=text+row['見出し']
        return len(text)
    
    def judge_countword(self,  word):
        len_word = len(word)
        if len_word>self.MIN_LEN and len_word<self.MAX_LEN:
          return True
        else:
          return False
    
    def judge_countword_out(self,  word):
        len_word = len(word)
        if len_word>self.MAX_LEN:
          return True
        else:
          return False
    
    def merge_df(self,df):
        """
        DataFrame受け取って文章にして返す
        """
        text=''
        for index,row in df.iterrows():
            text=text+row['見出し']
        return text
    
    
    def find_child(self, df,id,merge_text,id_list):
        """
        IDを受け取り、
        子文節２以上のところまで降っていく
        もし子文節２より小さいなら、親文節にあたるもののIDを返し、textにたす。
        2以上まできたら、textにはたさずに、そこまでで返す
        """
        #その行とりだし
        this_line = df[df['ID']==id].reset_index()
        this_line = this_line.iloc[0]



        #もし親文節ならそこでおわり
        if  this_line['親文節']==-1:
          return merge_text, this_line.loc['ID'],id_list

        #子文節が２つ以上か
        if this_line.loc['子文節個数']>1:
          return merge_text ,this_line.loc['ID'],id_list
        else:
          id_list = id_list[id_list['ID'] !=this_line.loc['ID']]
          merge_text = merge_text + this_line.loc['見出し']
          return self.find_child(df, this_line.loc['親文節'],merge_text,id_list)

    
    def child_root(self, df):
      return df[df['子文節個数']==0]

    def merge_root(self, df):
      merged_df = pd.DataFrame(columns=['ID','見出し','親文節','子文節個数'])
      id_list = pd.DataFrame(df['ID'])
      
      
      for index, row in df[df['子文節個数']==0].iterrows(): 
        merge_text=''
        merge_text, parent_id, id_list=self.find_child(df,row['ID'], merge_text, id_list)
        merged_df = merged_df.append({'ID':row['ID'], '見出し':merge_text, '親文節':parent_id,'子文節個数':0},ignore_index=True)
      return merged_df, id_list
    
    def merge_parent(self,df,child_df,left_id):
      """
      分岐が１つしかないような文節を整形した際、
      整形されなかった文節がleft_idに入っている。
      left_idに含まれた文節について、足す。
      整形された全ての文節を含むdataframeを返す
      """
      #comp_df= self.return_df()
      for index,row in left_id.iterrows():
        child_df = child_df.append(df[df['ID']==row['ID']])
      return child_df.sort_values('ID')
    
    def all_branch(self,parents_word, child_list,all_pattern_list):
      """
      その文節からの分岐が渡される
      全列挙する
      """
      all_branch_list = copy.copy(all_pattern_list)
      all_branch_list.append('')
      #全ての子文節について
      child_branch = copy.copy(all_branch_list)

      for child in child_list:
        #その文節があるかないか2つ
        for branch in child_branch:
          if not self.judge_countword_out(branch + child):
            all_branch_list.append(branch+child)
        child_branch = copy.copy(all_branch_list)
      all_branch_list = [i+parents_word for i in all_branch_list]
      
      return all_branch_list
      
    def judge_branch_list(self, branch_list):
      """
      候補文全てが入ったリストを受け取る
      長さを判定して、長さが不適切な要素は削除
      候補文のリストを返す
      """
      branch_list = [word for word in branch_list if len(word)>self.MIN_LEN and len(word)<self.MAX_LEN]
      return branch_list
    
    def return_all_search(self, df):
      """
      候補文を全列挙する
      """
      return_list=[]
      for index,row in df[df['子文節個数']>0].iterrows():
        child_df = df[(df['親文節']==row['ID']) & (df['子文節個数']==0 )]
        child_list = child_df['見出し'].tolist()
        return_list.extend(self.all_branch(row['見出し'], child_list,return_list))
        return_list = [word for word in return_list if not  len(word)>self.MAX_LEN]
      return return_list



            
def s_main(line):
    sc = SentenceCompression()

    root_text_list=[]
    #１行ずつ見る
    all_root_text_list = []
    df = sc.phrase_analysis(line)
    
    count_df = sc.count_childroot(df)

    cols = ['ID', '見出し','親文節','子文節個数']
    merge_df, left_id=sc.merge_root(count_df)
    merge_df=sc.merge_parent(count_df,merge_df,left_id)


    all_root_text_list= sc.return_all_search(merge_df)


    judge_root_text = sc.judge_branch_list(all_root_text_list)
    #root_text_list.append(judge_root_text)
        



        
        
    
    return (judge_root_text)


def wakati_jm(text):
    jumanpp = Juman(jumanpp=False)
    result = jumanpp.analysis(text)
    tokenized_text =[mrph.midasi for mrph in result.mrph_list()]
    return " ".join(tokenized_text)

def lex_rank(cand_text_list):
  #corpus = [wakati_jm(t) for t in cand_text_list]
  
  model = TfidfVectorizer()
  
  corpus = [i for i in cand_text_list if not len(i)<1]
  
  tfidf_vecs = model.fit_transform(corpus).toarray()
  cos_sim = cosine_similarity(tfidf_vecs, tfidf_vecs)

  thr = 0.3
  adjacency_matrix = np.zeros(cos_sim.shape)
  
  for i in range(len(cand_text_list)):
          for j in range(len(cand_text_list)):
              if cos_sim[i, j] > thr:
                  adjacency_matrix[i, j] = 1

  stochastic_matrix = adjacency_matrix
  for i in range(len(cand_text_list)):
      degree = stochastic_matrix[i, ].sum()
      if degree > 0:
          stochastic_matrix[i, ] /= degree

  lexrank_score = power_method(stochastic_matrix, 10e-6)

  top_score = np.argsort(lexrank_score)
  top_score = top_score[::-1][0]
  return_text = cand_text_list[top_score].replace(' ','')

  return return_text
  

def tf_idf(corpus, cand_list):
  """
  元テキストを使ってtf-idfの値の計算をし、
  各候補分で全て足し合わせる。もっとも値が大きい候補文を返す。
  """
  #lines = text.split('。')
  #jumanpp = Juman(jumanpp=False)  
  corpus = [wakati_jm(t) for t in lines]

  model = TfidfVectorizer()
  corpus = [i for i in corpus if not len(i)<1]
  tfidf_vecs = model.fit_transform(corpus).toarray()

  values = tfidf_vecs
  feature_name = model.get_feature_names()
  df = pd.DataFrame(values, columns = feature_name)


  tf_idf_df = pd.DataFrame()

  tf_idf_df = pd.DataFrame(cand_list,columns=['候補文'])
  tf_idf_df['TF-IDF'] = 0
  for index,row in tf_idf_df.iterrows():
    tf_idf_sum =0
    for column in df.columns:
      if column in row['候補文']:
        tf_idf_sum= tf_idf_sum+df.loc[0,column]


    tf_idf_df.loc[index,'TF-IDF']=tf_idf_sum
  if len(tf_idf_df)>0:
    return tf_idf_df.sort_values('TF-IDF', ascending=False).reset_index().loc[0,'候補文']
  else:
    return ''


def power_method(stochastic_matrix, epsilon):
    n = stochastic_matrix.shape[0]
    p = np.ones(n)/n
    delta = 1
    while delta > epsilon:
        _p = np.dot(stochastic_matrix.T, p)
        delta = np.linalg.norm(_p - p)
        p = _p
    return p

def return_corpus(ori_corpus, text):
    corpus_text = []
    while(True):
        for corpus in ori_corpus:
            if text.startswith(corpus):
                corpus_text.append(corpus)
                corpus_text.append(' ')
                text=text[(len(corpus)):]
        if len(text)==0:
            break
    return_text=''
    for i in corpus_text:
        return_text=return_text+i
    return return_text
        

text='正確な情報も出揃って広がってきているし引き下がらなくてもと思ったが、到底人間が耐えられるような状態じゃなかったのでなるほどという感じだ。'
lines = text.split('。')



for line in lines:
    if len(line)==0:
      continue

    sum_texts = s_main(line)
    print(len(sum_texts))



    ori_corpus = wakati_jm(line)
    ori_corpus = ori_corpus.split(' ')

    cand_corpus_list=[]
    for sum_text in sum_texts:
        corpus_text = return_corpus(ori_corpus, sum_text)
        cand_corpus_list.append(corpus_text)

    print(tf_idf(ori_corpus,sum_texts))
    print(lex_rank(cand_corpus_list))

    result = timeit.repeat('tf_idf(ori_corpus,sum_texts)', number=1,repeat=1,globals=globals())
    print(f'td_idfの実行時間:{result}')

    result = timeit.repeat('lex_rank(cand_corpus_list)', number=1,repeat=1,globals=globals())
    print(f'lexrankの実行時間:{result}')