

import os
import firebase_admin
from firebase_admin import credentials
os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', 'hudl_server/fbpk.json')

cred = credentials.Certificate('hudl_server/fbpk.json')
firebase_admin.initialize_app(cred)
