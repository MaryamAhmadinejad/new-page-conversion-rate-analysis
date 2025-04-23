import pandas as pd
import numpy as np


class DataFrameMemoryOptimizer:
    def __init__(self, df, verbose=False, cat_threshold=0.5):
        self.df = df.copy()
        self.verbose = verbose
        self.cat_threshold = cat_threshold
        self.start_mem = self._memory_usage()
        self.end_mem = None

    def _memory_usage(self):
        return self.df.memory_usage(deep=True).sum() / 1024**2  # MB

    def _optimize_numeric(self):
        for col in self.df.select_dtypes(include=["int", "float"]).columns:
            col_data = self.df[col]
            if pd.api.types.is_integer_dtype(col_data):
                self.df[col] = pd.to_numeric(col_data, downcast='unsigned' if col_data.min() >= 0 else 'signed')
            elif pd.api.types.is_float_dtype(col_data):
                self.df[col] = pd.to_numeric(col_data, downcast='float')

    def _optimize_object(self):
        for col in self.df.select_dtypes(include=["object"]).columns:
            num_unique = self.df[col].nunique()
            num_total = len(self.df[col])
            if num_unique / num_total < self.cat_threshold:
                self.df[col] = self.df[col].astype("category")

    def optimize(self):
        self._optimize_numeric()
        self._optimize_object()
        self.end_mem = self._memory_usage()
        if self.verbose:
            self.report()
        return self.df

    def report(self):
        if self.end_mem is None:
            self.end_mem = self._memory_usage()
        print('#'*50)
        print(f"Memory usage decreased from {self.start_mem:.2f} MB to {self.end_mem:.2f} MB")
        print(f"Decreased by {(100 * (self.start_mem - self.end_mem) / self.start_mem):.1f}%")
        print('#'*50)
        print(self.df.dtypes)
        print('#'*50)

