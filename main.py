import os
import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import anthropic
import stripe
import httpx
from bs4 import BeautifulSoup
from pymongo import MongoClient

load_dotenv()
