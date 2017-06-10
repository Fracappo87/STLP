from setuptools import setup

setup(name='STLP_py',
      version='0.1',
      description='Statistical tools for lazy people',
      url='https://github.com/Fracappo87/STPL',
      download_url='https://github.com/Fracappo87/STPL.git',
      author='Francesco Capponi',
      author_email='capponi.francesco87@gmail.com',
      license='BSD 3 clause',
      #packages=['boruta'],
      #package_dir={'boruta': 'boruta'},
      #package_data={'boruta/examples/*csv': ['boruta/examples/*.csv']},
      #include_package_data = True,
      keywords=['Nan-percentage', 'shapes detection', 'ESD method','outliers','outliers detection'],
      install_requires=['numpy>=1.10.4',
                        'scikit-learn>=0.17.1',
                        'pandas>=0.17.0'
                        ])
