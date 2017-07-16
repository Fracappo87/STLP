import unittest
from STLP_py import MissingData
import pandas as pd
import numpy as np

""" Test data frame

     a  b    c    d
0  NaN  5    2  NaN
1  NaN  5  NaN  NaN
2  NaN  5    2    3

0.<treshold<=0.33 => all the columns with NaN are considered as relevant
0.33<treshold<0.7 => only first and fourth columns with NaN are considered as relevant
0.7<treshold <1 => only the first column with NaN is considered as relevant

"""

class MissingDataTest(unittest.TestCase):

    def test_support_features(self):
        df = pd.DataFrame(columns=['a','b','c','d'], index=['0','1','2'])
        df.loc['0'] = pd.Series({'a':np.nan, 'b':5, 'c':2, 'd':np.nan})
        df.loc['1'] = pd.Series({'a':np.nan, 'b':5, 'c':np.nan, 'd':np.nan})
        df.loc['2'] = pd.Series({'a':np.nan, 'b':5, 'c':2, 'd':3})

        mo = MissingData()
        tro=[.1,.4,.8]
        supp=['a','d','c']
        nfeat=3
        suppfi=[['a','d','c'],['a','d'],['a']]
        nfeatfi=[3,2,1]

        for i,j,k in zip(tro,suppfi,nfeatfi):

            mo.count(df,['a','b','c','d'],i)
            self.assertEqual(mo.support_,supp,"Checking global support")
            self.assertEqual(mo.n_features_,nfeat,"Checking number of features in global support")

            self.assertEqual(mo.support_filter_,j,"Checking filtered support")
            self.assertEqual(mo.n_features_filter_,k,"Checking number of features in filtered support")

    def test_transform(self):
        df = pd.DataFrame(columns=['a','b','c','d'], index=['0','1','2'])
        df.loc['0'] = pd.Series({'a':np.nan, 'b':5, 'c':2, 'd':np.nan})
        df.loc['1'] = pd.Series({'a':np.nan, 'b':5, 'c':np.nan, 'd':np.nan})
        df.loc['2'] = pd.Series({'a':np.nan, 'b':5, 'c':2, 'd':3})

        mo = MissingData()
        tro=[.1,.4,.8]
        dfs=[df.drop(['a','c','d'],axis=1),df.drop(['a','d'],axis=1),df.drop('a',axis=1)]

        for i,j in zip(tro,dfs):

            mo.count(df,['a','b','c','d'],i)
            self.assertEqual(mo.transform(df).shape,j.shape,"Checking transformed data frame shape")
            self.assertEqual(mo.transform(df).columns.all(),j.columns.all(),"Checking transformed data frame columns")

            for k in mo.transform(df).columns:
                self.assertEqual(mo.transform(df)[k].all(),j[k].all(),"Checking transformed data frame data")


if __name__ == '__main__':
    unittest.main()
