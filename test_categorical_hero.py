"""
Author: Francesco Capponi <capponi.francesco87@gmail.com>

License: BSD 3 clause

"""



import unittest
from STLP_py import CategoricalHero
import pandas as pd
import numpy as np



"""

Test data frames

TRAIN
a      b       c
0  car    cat     neo
1  truck  dog     morpheus
2  NaN    cheetah trinity
3  bike   lion    NaN

TEST
a        b       c
0  car      cat     neo
1  truck    dog     morpheus
2  tractor  lion    trinity
3  bike     lion    morpheus

TRAIN: feature b has category "cheetah", missing in TEST
TEST: feature a has category "tractor", missing in TRAIN

"""

class CategoricalHeroTest1(unittest.TestCase):

    def test_detection(self):

        train = pd.DataFrame(columns=['a','b','c'], index=['0','1','2','3'])
        train.loc['0'] = pd.Series({'a':'car', 'b':'cat', 'c':'neo'})
        train.loc['1'] = pd.Series({'a':'truck', 'b':'dog', 'c':'morpheus'})
        train.loc['2'] = pd.Series({'a':np.nan, 'b':'cheetah', 'c':'trinity'})
        train.loc['3'] = pd.Series({'a':'bike', 'b':'lion', 'c':np.nan})

        test = pd.DataFrame(columns=['a','b','c'], index=['0','1','2','3'])
        test.loc['0'] = pd.Series({'a':'car', 'b':'cat', 'c':'neo'})
        test.loc['1'] = pd.Series({'a':'truck', 'b':'dog', 'c':'morpheus'})
        test.loc['2'] = pd.Series({'a':'tractor', 'b':'lion', 'c':'trinity'})
        test.loc['3'] = pd.Series({'a':'bike', 'b':'lion', 'c':'morpheus'})

        mo = CategoricalHero()
        mo.shape_detector(train,test,['a','b','c'])

        check_detected=['a','b']
        self.assertEqual(mo.detected_,check_detected,"Checking detected features")


        checka=pd.DataFrame(columns=["train" ,"test"], index=['bike','car','tractor','truck','% missings'])
        checka.columns = pd.MultiIndex.from_tuples(list(zip(checka.columns, ["a", "a"])))
        checka.loc['bike'] = pd.Series({('train','a'):1./3.,('test','a'):1./4.})
        checka.loc['car'] = pd.Series({('train','a'):1./3.,('test','a'):1./4.})
        checka.loc['tractor'] = pd.Series({('train','a'):np.nan,('test','a'):1./4.})
        checka.loc['truck'] = pd.Series({('train','a'):1./3.,('test','a'):1./4.})
        checka.loc['% missings'] = pd.Series({('train','a'):1./4.,('test','a'):0.0})


        checkb=pd.DataFrame(columns=['train','test'], index=['cat','cheetah','dog','lion','% missings'])
        checkb.columns = pd.MultiIndex.from_tuples(list(zip(checkb.columns, ["b", "b"])))
        checkb.loc['cat'] = pd.Series({('train','b'):1./4.,('test','b'):1./4.})
        checkb.loc['cheetah'] = pd.Series({('train','b'):1./4.,('test','b'):np.nan})
        checkb.loc['dog'] = pd.Series({('train','b'):1./4.,('test','b'):1./4.})
        checkb.loc['lion'] = pd.Series({('train','b'):1./4.,('test','b'):1./2.})
        checkb.loc['% missings'] = pd.Series({('train','b'):0.0,('test','b'):0.0})

        check={'a':checka,'b':checkb}


        for i in mo.detected_:
            self.assertEqual(mo.detector_[i].shape,check[i].shape,"Checking detector data frame shape")
            self.assertEqual(len(mo.detector_[i].columns),len(check[i].columns),"Checking multiindex length")
            for j in range(len(mo.detector_[i].columns)):
                self.assertEqual(mo.detector_[i].columns[j],check[i].columns[j],"Checking detector data frame columns")
                colA=mo.detector_[i].columns[j]
                colB=check[i].columns[j]
                for k,j in zip(mo.detector_[i][colA],check[i][colB]):
                    np.testing.assert_equal(k,j,"Checking detector data frame data")

if __name__ == '__main__':
    unittest.main()
