from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from pydantic import BaseModel
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pickle
import sklearn
import fastapi
import python_multipart
import uvicorn

print("="*10)
print(f"sklearn: {sklearn.__version__}")
print(f"fastapi: {fastapi.__version__}")
print(f"python_multipart: {python_multipart.__version__}")
print(f"uvicorn: {uvicorn.__version__}")

