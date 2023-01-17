{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a925df9c-4ac1-4489-9684-51bf20d05f85",
   "metadata": {},
   "source": [
    "## Running a NLP Project on Gilgamesh"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "b9cfb711-58b8-4872-bd6c-111ea9182e0b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Files in '/Users/julianlavecchia/Desktop/NLP_PracticeWork': ['Gilgamesh_Tablet_I.txt', 'README.md', '.ipynb_checkpoints', '.git', 'NLP_Gilgamesh.ipynb', 'Gilgamesh_Table_I_clean.txt']\n"
     ]
    }
   ],
   "source": [
    "# Asses the filepath of the txt file\n",
    "\n",
    "import os\n",
    "\n",
    "# Get the current working directory (cwd)\n",
    "cwd = os.getcwd()\n",
    "\n",
    "# Get all the files in that directory\n",
    "files = os.listdir(cwd)\n",
    "\n",
    "print(\"Files in %r: %s\" % (cwd, files))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "65afc20a-fbbc-4cf6-a112-8383873d1df1",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(\"Gilgamesh_Tablet_I.txt\", \"r\") as f:\n",
    "    \n",
    "    # Instantiate an empty list\n",
    "    new_lines=[]\n",
    "    \n",
    "    # Set up for-loop to iterate and append lines into list\n",
    "    for idx, line in enumerate(f):\n",
    "        new_lines.append(line)\n",
    "        \n",
    "with open(\"Gilgamesh_Table_I_clean.txt\", \"w\") as f:\n",
    "    \n",
    "    # Set up for-loop to write lines into list\n",
    "    for line in new_lines:\n",
    "        f.write(line)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba76623e-0417-46a3-9f20-d04a29676212",
   "metadata": {},
   "source": [
    "### 1. Load the Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ed57ed73-84f9-4cdb-89ab-ee601f160857",
   "metadata": {},
   "outputs": [],
   "source": [
    "# load text\n",
    "filename = 'Gilgamesh_Tablet_I.txt'\n",
    "\n",
    "# open file\n",
    "file = open(filename, 'rt')\n",
    "\n",
    "# set variable so file is saved to memory\n",
    "text = file.read()\n",
    "file.close()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4ce40742-e250-4d06-9579-46d04cdc9e75",
   "metadata": {},
   "source": [
    "### 2. Cleaning Text Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "699e56ec-8162-4052-8f32-a9486d10f567",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'str'>\n"
     ]
    }
   ],
   "source": [
    "# Print their type\n",
    "print(type(text))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6ccfe89a-f713-4e6c-ba5c-53e6e9d105b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "\n",
    "# Get rid of all words between square parentheses\n",
    "text = re.sub(\"\\[.*?\\]\",\"\",text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "eb0ffbff-d45f-48ed-ad29-cb45a0991ffc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['seen', 'everyth', 'make', 'known', 'land', 'teach', 'experienc', 'thing', 'alik', 'anu', 'grant', 'total', 'knowledg', 'saw', 'secret', 'discov', 'hidden', 'brought', 'inform', 'time', 'flood', 'went', 'distant', 'journey', 'push', 'exhaust', 'brought', 'peac', 'carv', 'stone', 'stela', 'toil', 'built', 'wall', 'wall', 'sacr', 'eanna', 'templ', 'holi', 'sanctuari', 'look', 'wall', 'gleam', 'like', 'copper', 'inspect', 'inner', 'wall', 'like', 'one', 'equal', 'take', 'hold', 'threshold', 'stone', 'date', 'ancient', 'time', 'go', 'close', 'eanna', 'templ', 'resid', 'ishtar', 'later', 'king', 'man', 'ever', 'equal', 'go', 'wall', 'uruk', 'walk', 'around', 'examin', 'foundat', 'inspect', 'brickwork', 'thoroughli', 'even', 'core', 'brick', 'structur', 'made', 'brick', 'seven', 'sage', 'lay', 'plan', 'one', 'leagu', 'citi', 'one', 'leagu', 'palm', 'garden', 'one', 'leagu', 'lowland', 'open']\n"
     ]
    }
   ],
   "source": [
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "\n",
    "# Tokenize text\n",
    "tokens = word_tokenize(text)\n",
    "\n",
    "# Convert to lower case\n",
    "tokens = [token.lower() for token in tokens]\n",
    "\n",
    "# Remove all tokens that are not alphabetic\n",
    "words = [word for word in tokens if word.isalpha()]\n",
    "\n",
    "# Instantiate a variable containing all english stopwords\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "# Iterate over words for all words that are not in stop_words\n",
    "words = [w for w in words if not w in stop_words]\n",
    "\n",
    "# Instantiate Porter stemmer\n",
    "porter = PorterStemmer()\n",
    "stemmed = [porter.stem(word) for word in words]\n",
    "\n",
    "print(stemmed[:100])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2b58af99-e721-4845-b7eb-b0cbdc4b4e81",
   "metadata": {},
   "source": [
    "## Creating a Word2Vec Model of Tablet I Gilgamesh"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cde1ee8b-f792-4c34-89df-98508f993f1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "all_words = []\n",
    "\n",
    "all_words.append(stemmed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f431db70-df84-4a56-881f-c1c12511d807",
   "metadata": {},
   "outputs": [],
   "source": [
    "# A value of 2 for min_count specifies to include only those words in the Word2Vec\n",
    "# model that appear at least twice in the corpus.\n",
    "\n",
    "from gensim.models import Word2Vec\n",
    "\n",
    "word2vec = Word2Vec(all_words, min_count=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "fc9b5ac3-8612-4345-8aed-06c4cddf38af",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['gilgamesh',\n",
       " 'anim',\n",
       " 'water',\n",
       " 'uruk',\n",
       " 'enkidu',\n",
       " 'anu',\n",
       " 'mighti',\n",
       " 'wild',\n",
       " 'like',\n",
       " 'harlot']"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Display unique words that appear at least twice in the dictionary\n",
    "\n",
    "vocabulary = word2vec.wv.index_to_key\n",
    "vocabulary[:10]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "02ab03c3-a856-4890-b27f-f37ad5cf1275",
   "metadata": {},
   "source": [
    "## Model Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "d04070da-801b-444b-a430-a2fa1c7a20bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[-0.00097155  0.00092604  0.00505691  0.00898258 -0.00904752 -0.00853458\n",
      "  0.00710687  0.01116475 -0.00544173 -0.00410399  0.00735215 -0.00252616\n",
      " -0.00422281  0.0071439  -0.00513358 -0.0018103   0.00341722 -0.00030664\n",
      " -0.00848165 -0.01163833  0.00717137  0.00544011  0.00719393  0.00055877\n",
      "  0.00555216 -0.00321793 -0.00124406  0.00464348 -0.00824599 -0.00387258\n",
      " -0.00688998 -0.00027598  0.01001161 -0.00760805 -0.00316106 -0.00013775\n",
      "  0.00830277 -0.00677509 -0.00067253 -0.00625716 -0.00991809  0.00398214\n",
      " -0.00891244 -0.00459525  0.0009727   0.00011903 -0.00796571  0.00927949\n",
      "  0.00525213  0.00993312 -0.00811783  0.00422281 -0.00416856  0.00087408\n",
      "  0.00850764 -0.00417343  0.0044395  -0.00700492 -0.00442927  0.00980315\n",
      " -0.00155761  0.00067804 -0.0039007  -0.00822894 -0.00259689  0.00295146\n",
      " -0.00013248  0.00663394 -0.00374279  0.00382101  0.00492422  0.00888534\n",
      " -0.00116408 -0.00976185  0.00540907  0.00111771  0.00784079 -0.00083287\n",
      " -0.00385229 -0.00865246 -0.00092005  0.00345917  0.00442339  0.00864447\n",
      " -0.00667656  0.00178744  0.00683788 -0.00341679 -0.00222219  0.00637562\n",
      "  0.00293316  0.00087704  0.00359395  0.00095447  0.01140459  0.00595662\n",
      " -0.00816504 -0.00769563  0.00114176  0.00594997]\n",
      "[('night', 0.37048977613449097), ('meteorit', 0.3249882757663727), ('drew', 0.2994292676448822), ('open', 0.2761461138725281), ('warrior', 0.26526936888694763), ('tri', 0.24756883084774017), ('complaint', 0.23217898607254028), ('feet', 0.20535792410373688), ('one', 0.1878046840429306), ('daughter', 0.18750306963920593)]\n"
     ]
    }
   ],
   "source": [
    "print(word2vec.wv['gilgamesh'])\n",
    "print(word2vec.wv.most_similar('mighti'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d55108f-3120-40f5-8652-d71b169ad772",
   "metadata": {},
   "source": [
    "## Visualizing Word Similarity with T-SNE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "f8f0cbd3-202b-4f1a-b6b6-40d5bf2ff2d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def display_closestwords_tsnescatterplot(word2vec, word):\n",
    "    \n",
    "    arr = np.empty((0,100), dtype='f')\n",
    "    word_labels = [word]\n",
    "\n",
    "    # get close words\n",
    "    close_words = word2vec.wv.most_similar(word)\n",
    "    \n",
    "    # add the vector for each of the closest words to the array\n",
    "    arr = np.append(arr, np.array([word2vec.wv[word]]), axis=0)\n",
    "    for wrd_score in close_words:\n",
    "        wrd_vector = word2vec.wv[wrd_score[0]]\n",
    "        word_labels.append(wrd_score[0])\n",
    "        arr = np.append(arr, np.array([wrd_vector]), axis=0)\n",
    "        \n",
    "    # find tsne coords for 2 dimensions\n",
    "    tsne = TSNE(n_components=2, random_state=0, perplexity=5)\n",
    "    np.set_printoptions(suppress=True)\n",
    "    Y = tsne.fit_transform(arr)\n",
    "\n",
    "    x_coords = Y[:, 0]\n",
    "    y_coords = Y[:, 1]\n",
    "    # display scatter plot\n",
    "    plt.scatter(x_coords, y_coords)\n",
    "\n",
    "    for label, x, y in zip(word_labels, x_coords, y_coords):\n",
    "        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')\n",
    "    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)\n",
    "    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)\n",
    "    plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "d752914e-3438-4740-86a1-8728f4290d41",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/julianlavecchia/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_t_sne.py:800: FutureWarning: The default initialization in TSNE will change from 'random' to 'pca' in 1.2.\n",
      "  warnings.warn(\n",
      "/Users/julianlavecchia/opt/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_t_sne.py:810: FutureWarning: The default learning rate in TSNE will change from 200.0 to 'auto' in 1.2.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "image/png": "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UIAjR45g/fr1iIqKwpw5c6qibCIiIqInYvBAlZubiyFDhuDLL79EvXr1pPk5OTn46quvsHTpUnTr1g3+/v5Yt24djhw5gqNHjwIA9uzZg9TUVHzzzTfw9fXFyy+/jPnz5+OLL75AQUGBoUsnIiIieiIGD1RhYWHo3bs3goKC9OYnJSXh3r17evNbtWoFNzc3xMfHAwDi4+Ph4+MDZ2dnqU1wcDB0Oh3Onj1b7j7z8/Oh0+n0PkRERESGYmLIjW/cuBEnTpzAsWPHSi3TarUwMzODnZ2d3nxnZ2dotVqpzYNhqmR5ybLyREREIDw8XGb1RERERE/GYGeoLl++jIkTJ+Lbb7+FhYWFoXZTppkzZyInJ0f6XL58uUr3T0RERHWLwQJVUlISsrKy4OfnBxMTE5iYmCAuLg6ffvopTExM4OzsjIKCAty8eVNvvczMTKjVagCAWq0u9dRfyXRJm7KYm5tDqVTqfYiIiIgMxWCBqnv37jhz5gySk5OlT7t27TBkyBDpZ1NTU+zbt09aJy0tDenp6dBoNAAAjUaDM2fOICsrS2qzd+9eKJVKeHl5Gap0IiIiogox2D1Utra28Pb21ptnbW2N+vXrS/NDQ0MxefJk2NvbQ6lUYsKECdBoNOjUqRMAoEePHvDy8sKwYcOwcOFCaLVazJ49G2FhYTA3NzdU6UREREQVYtCb0h9n2bJlMDIywoABA5Cfn4/g4GCsXLlSWm5sbIwdO3Zg3Lhx0Gg0sLa2xogRI/DRRx9VY9VERERE+hRCCFHdRRiaTqeDSqVCTk4O76ciIiKqJWrT9zff5UdEREQkEwMVERERkUwMVEREREQyMVARERERycRARURERCQTAxURERGRTAxURERERDIxUBERERHJxEBFREREJBMDFREREZFMDFREREREMjFQEREREcnEQEVEREQkEwMVERERkUwMVEREREQyMVARERERycRARURERCQTAxURERGRTAxURERERDIxUBERERHJxEBFREREJBMDFREREZFMDFREREREMjFQEREREcnEQEVEREQkEwMVERERkUwMVEREREQyMVARERERycRARURERCQTAxURERGRTAxURERERDIxUBERERHJxEBFREREJBMDFREREZFMDFREREREMhk0UEVGRqJ169ZQKpVQKpXQaDTYtWuXtDwvLw9hYWGoX78+bGxs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P/+MkJAQAPfPTjVo0ABTpkzB1KlTAQA5OTlwdnZGVFQUBg0ahHPnzsHLywvHjh1Du3btAAAxMTHo1asXrly5ggYNGjzRvnU6HVQqFXJycqBUKg3SPyIiIqpcten7u9ruobp48SK0Wi2CgoKkeSqVCh07dkR8fDwAID4+HnZ2dlKYAoCgoCAYGRkhISGh3G3n5+dDp9PpfYiIiIgMpdoClVarBQA4OzvrzXd2dpaWabVaODk56S03MTGBvb291KYsERERUKlU0sfV1bWSqyciIiL6n2fyKb+ZM2ciJydH+ly+fLm6SyIiIqJnWLUFKrVaDQDIzMzUm5+ZmSktU6vVyMrK0lteWFiI7OxsqU1ZzM3NoVQq9T5EREREhlJtgapJkyZQq9XYt2+fNE+n0yEhIQEajQYAoNFocPPmTSQlJUlt9u/fj+LiYnTs2LHKayYiIiIqi4khN56bm4vff/9dmr548SKSk5Nhb28PNzc3TJo0Cf/85z/RokULNGnSBB9++CEaNGggPQno6emJnj17YtSoUVi1ahXu3buH8ePHY9CgQU/8hB8RERGRoRk0UB0/fhxdu3aVpidPngwAGDFiBKKiovD+++/j9u3bGD16NG7evInnn38eMTExsLCwkNb59ttvMX78eHTv3h1GRkYYMGAAPv30U0OWTURERFQhVTYOVXWqTeNYEBER0X216fv7mXzKj4iIiKgqMVARERERycRARURERCQTAxURERGRTAxURERERDIxUBERERHJxEBFREREJBMDFREREZFMDFREREREMjFQEREREcnEQEVEREQkEwMVERERkUwMVEREREQyMVARERERycRARURERCQTAxURERGRTAxURERERDIxUBERERHJxEBFREREJBMDFREREZFMDFREREREMjFQEREREcnEQEVEREQkEwMVERERkUwMVEREREQyMVARERERycRARURERCQTAxURERGRTAxURERERDIxUBERERHJxEBFREREJBMDFREREZFMDFREREREMjFQEREREclUawLVF198AXd3d1hYWKBjx45ITEys7pKIiIiIANSSQPX9999j8uTJmDt3Lk6cOIE2bdogODgYWVlZ1V0aERERUe0IVEuXLsWoUaPw1ltvwcvLC6tWrYKVlRX+/e9/V3dpRERERDU/UBUUFCApKQlBQUHSPCMjIwQFBSE+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Ca1atYKFhQV8fHwQHR1tqLKJiGoEYyMF5va9/7Tew6GqZHpuXy8YG5UXuYioKhksUJ0/fx7FxcVYvXo1zp49i2XLlmHVqlX44IMPpDY6nQ49evRA48aNkZSUhEWLFmHevHlYs2aN1ObIkSMYPHgwQkNDcfLkSYSEhCAkJAQpKSmGKp2IqEbo6e2CyKF+UKv0L+upVRaIHOqHnt4u1VQZET2sSm9KX7RoESIjI/HHH38AACIjIzFr1ixotVqYmZkBAGbMmIEtW7bg/PnzAICBAwfi9u3b2LFjh7SdTp06wdfXF6tWrXqi/damm9qIiB724EjpTrb3L/PxzBTVBbXp+7tK76HKycmBvf3/rvfHx8fjhRdekMIUAAQHByMtLQ03btyQ2gQFBeltJzg4GPHx8eXuJz8/HzqdTu9DRFRbGRspoGlWH6/6NoSmWX2GKaIaqMoC1e+//47PPvsMY8aMkeZptVo4OzvrtSuZ1mq1j2xTsrwsERERUKlU0sfV1bWyukFERERUSoUD1YwZM6BQKB75KblcV+Lq1avo2bMnXn/9dYwaNarSii/PzJkzkZOTI30uX75s8H0SERFR3VXhkdKnTJlS7ii7JR58W/m1a9fQtWtXBAQE6N1sDgBqtRqZmZl680qm1Wr1I9uULC+Lubk5zM3NH9sXIiIiospQ4UDl6OgIR0fHJ2p79epVdO3aFf7+/li3bh2MjPRPiGk0GsyaNQv37t2TXpWwd+9eeHh4oF69elKbffv26b1Xau/evdBoNBUtnYiIiMggDHYP1dWrVxEYGAg3NzcsXrwYf/31F7Rard69T2+++SbMzMwQGhqKs2fP4vvvv8eKFSswefJkqc3EiRMRExODJUuW4Pz585g3bx6OHz+O8ePHG6p0IiIiogox2MuR9+7di99//x2///47GjVqpLesZKQGlUqFPXv2ICwsDP7+/nBwcMCcOXMwevRoqW1AQAA2bNiA2bNn44MPPkCLFi2wZcsWeHt7G6p0IiIiogrhy5GJiIioRqpN3998lx8RERGRTAxURERERDIxUBERERHJxEBFREREJBMDFREREZFMDFREREREMjFQEREREcnEQEVEREQkEwMVERERkUwMVEREREQyMVARERERycRARURERCQTAxURERGRTAxURERERDIxUBERERHJxEBFREREJBMDFREREZFMDFRE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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "display_closestwords_tsnescatterplot(word2vec, 'water')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd8966c1-c88f-4c09-b9d0-f6aea0bc02b7",
   "metadata": {},
   "source": [
    "## Word Counts with CountVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "ea1e8b59-cad3-4e5f-a99f-db1eb1dbe1e2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "vocabulary:  {'seen': 342, 'everyth': 123, 'make': 250, 'known': 217, 'land': 220, 'teach': 384, 'experienc': 126, 'thing': 388, 'alik': 6, 'anu': 10, 'grant': 172, 'total': 401, 'knowledg': 216, 'saw': 335, 'secret': 339, 'discov': 95, 'hidden': 191, 'brought': 49, 'inform': 199, 'time': 398, 'flood': 147, 'went': 429, 'distant': 97, 'journey': 207, 'push': 307, 'exhaust': 125, 'peac': 287, 'carv': 55, 'stone': 369, 'stela': 368, 'toil': 399, 'built': 51, 'wall': 421, 'sacr': 327, 'eanna': 106, 'templ': 387, 'holi': 194, 'sanctuari': 330, 'look': 241, 'gleam': 164, 'like': 235, 'copper': 76, 'inspect': 202, 'inner': 200, 'one': 279, 'equal': 119, 'take': 382, 'hold': 192, 'threshold': 395, 'date': 85, 'ancient': 7, 'go': 165, 'close': 64, 'resid': 317, 'ishtar': 205, 'later': 223, 'king': 210, 'man': 251, 'ever': 121, 'uruk': 413, 'walk': 420, 'around': 16, 'examin': 124, 'foundat': 150, 'brickwork': 44, 'thoroughli': 391, 'even': 120, 'core': 77, 'brick': 43, 'structur': 376, 'made': 249, 'seven': 345, 'sage': 328, 'lay': 225, 'plan': 294, 'leagu': 229, 'citi': 61, 'palm': 285, 'garden': 156, 'lowland': 245, 'open': 280, 'area': 14, 'three': 394, 'enclos': 111, 'find': 143, 'tablet': 381, 'box': 42, 'lock': 239, 'bronz': 48, 'undo': 411, 'fasten': 133, 'read': 311, 'lapi': 221, 'lazuli': 226, 'gilgamesh': 161, 'everi': 122, 'hardship': 182, 'suprem': 380, 'lordli': 243, 'appear': 12, 'hero': 190, 'born': 40, 'gore': 170, 'wild': 433, 'bull': 52, 'front': 153, 'leader': 228, 'rear': 312, 'trust': 405, 'companion': 70, 'mighti': 260, 'net': 272, 'protector': 306, 'peopl': 288, 'rage': 308, 'destroy': 92, 'offspr': 278, 'lugalbanda': 246, 'strong': 373, 'perfect': 289, 'son': 360, 'august': 26, 'cow': 81, 'awesom': 29, 'mountain': 267, 'pass': 286, 'dug': 105, 'well': 428, 'flank': 146, 'cross': 83, 'ocean': 277, 'vast': 418, 'sea': 337, 'rise': 322, 'sun': 379, 'explor': 127, 'world': 441, 'region': 314, 'seek': 341, 'life': 233, 'reach': 310, 'sheer': 349, 'strength': 372, 'utanapishtim': 415, 'faraway': 132, 'restor': 318, 'teem': 385, 'mankind': 252, 'compar': 71, 'kingli': 211, 'say': 336, 'whose': 431, 'name': 269, 'day': 87, 'birth': 37, 'call': 53, 'god': 166, 'human': 196, 'great': 175, 'goddess': 167, 'design': 91, 'model': 265, 'bodi': 38, 'prepar': 301, 'form': 148, 'beauti': 34, 'handsomest': 181, 'men': 256, 'enclosur': 112, 'head': 184, 'rais': 309, 'other': 283, 'rival': 323, 'weapon': 427, 'fellow': 140, 'stand': 364, 'alert': 4, 'attent': 25, 'order': 282, 'becom': 35, 'anxiou': 11, 'leav': 230, 'father': 134, 'night': 274, 'arrog': 19, 'shepherd': 351, 'bold': 39, 'emin': 110, 'know': 215, 'wise': 435, 'girl': 162, 'mother': 266, 'daughter': 86, 'warrior': 423, 'bride': 45, 'young': 444, 'kept': 208, 'hear': 185, 'complaint': 73, 'heaven': 188, 'implor': 197, 'lord': 242, 'inde': 198, 'arrogantli': 20, 'listen': 236, 'aruru': 21, 'creat': 82, 'zikru': 447, 'let': 231, 'stormi': 371, 'heart': 187, 'match': 254, 'may': 255, 'heard': 186, 'within': 436, 'zikrtt': 446, 'wash': 424, 'hand': 179, 'pinch': 291, 'clay': 62, 'threw': 396, 'wilder': 434, 'valiant': 417, 'enkidu': 115, 'silenc': 354, 'endow': 113, 'ninurta': 275, 'whole': 430, 'shaggi': 347, 'hair': 178, 'full': 154, 'woman': 438, 'billow': 36, 'profus': 304, 'ashnan': 22, 'knew': 214, 'neither': 271, 'settl': 344, 'live': 238, 'wore': 440, 'garment': 157, 'sumukan': 378, 'ate': 24, 'grass': 174, 'gazel': 160, 'jostl': 206, 'water': 425, 'hole': 193, 'anim': 8, 'thirst': 390, 'slake': 358, 'mere': 257, 'notori': 276, 'trapper': 403, 'came': 54, 'opposit': 281, 'first': 145, 'second': 338, 'third': 389, 'see': 340, 'face': 131, 'stark': 366, 'fear': 136, 'drew': 102, 'back': 32, 'home': 195, 'rigid': 321, 'though': 392, 'pound': 299, 'drain': 98, 'color': 67, 'miser': 264, 'long': 240, 'address': 0, 'certain': 56, 'come': 68, 'mightiest': 262, 'meteorit': 258, 'continu': 75, 'goe': 168, 'place': 293, 'plant': 295, 'feet': 138, 'afraid': 3, 'fill': 142, 'pit': 292, 'wrench': 442, 'trap': 402, 'spread': 363, 'releas': 315, 'grasp': 173, 'round': 325, 'spoke': 362, 'stronger': 374, 'set': 343, 'tell': 386, 'might': 259, 'give': 163, 'harlot': 183, 'shamhat': 348, 'overcom': 284, 'drink': 103, 'robe': 324, 'expos': 128, 'sex': 346, 'draw': 100, 'near': 270, 'grew': 176, 'alien': 5, 'heed': 189, 'advic': 1, 'stood': 370, 'insid': 201, 'declar': 88, 'said': 329, 'bring': 46, 'direct': 94, 'way': 426, 'arriv': 18, 'appoint': 13, 'sat': 331, 'post': 298, 'drank': 99, 'beast': 33, 'eat': 108, 'primit': 303, 'savag': 333, 'depth': 90, 'clench': 63, 'arm': 15, 'voluptu': 419, 'restrain': 319, 'energi': 114, 'lie': 232, 'upon': 412, 'perform': 290, 'task': 383, 'womankind': 439, 'lust': 247, 'groan': 177, 'unclutch': 409, 'bosom': 41, 'took': 400, 'six': 355, 'stay': 367, 'arous': 17, 'intercours': 203, 'sate': 332, 'charm': 60, 'turn': 406, 'dart': 84, 'distanc': 96, 'utterli': 416, 'deplet': 89, 'knee': 213, 'want': 422, 'diminish': 93, 'run': 326, 'understand': 410, 'broaden': 47, 'gaze': 159, 'ear': 107, 'gallop': 155, 'strut': 377, 'power': 300, 'found': 149, 'favor': 135, 'awar': 27, 'sought': 361, 'friend': 152, 'away': 28, 'challeng': 57, 'shout': 352, 'lead': 227, 'chang': 59, 'us': 414, 'show': 353, 'skirt': 356, 'fineri': 144, 'festiv': 141, 'lyre': 248, 'drum': 104, 'play': 296, 'prettili': 302, 'exud': 130, 'laughter': 224, 'couch': 78, 'sheet': 350, 'extrem': 129, 'feel': 137, 'handsom': 180, 'youth': 445, 'fresh': 151, 'entir': 118, 'mightier': 261, 'without': 437, 'sleep': 359, 'wrong': 443, 'thought': 393, 'must': 268, 'love': 244, 'enlil': 117, 'la': 218, 'enlarg': 116, 'mind': 263, 'dream': 101, 'got': 171, 'reveal': 320, 'last': 222, 'star': 365, 'sky': 357, 'kind': 209, 'fell': 139, 'next': 273, 'tri': 404, 'lift': 234, 'could': 79, 'budg': 50, 'assembl': 23, 'populac': 297, 'throng': 397, 'cluster': 65, 'kiss': 212, 'littl': 237, 'babi': 31, 'embrac': 109, 'wife': 432, 'laid': 219, 'compet': 72, 'unabl': 408, 'comrad': 74, 'save': 334, 'strongest': 375, 'repeatedli': 316, 'good': 169, 'propiti': 305, 'anoth': 9, 'gate': 158, 'marit': 253, 'chamber': 58, 'axe': 30, 'collect': 66, 'command': 69, 'counselor': 80, 'advis': 2, 'interpret': 204, 'recount': 313, 'two': 407}\n",
      "shape:  (1099, 448)\n",
      "vectors:  [[0 0 0 ... 0 0 0]\n",
      " [0 0 0 ... 0 0 0]\n",
      " [0 0 0 ... 0 0 0]\n",
      " ...\n",
      " [0 0 0 ... 0 0 0]\n",
      " [0 0 0 ... 0 0 0]\n",
      " [0 0 0 ... 0 0 0]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "\n",
    "# create the transform\n",
    "vectorizer = CountVectorizer()\n",
    "\n",
    "# tokenize and build vocab\n",
    "vectorizer.fit(stemmed)\n",
    "\n",
    "print('vocabulary: ', vectorizer.vocabulary_)\n",
    "\n",
    "# encode document\n",
    "vector = vectorizer.transform(stemmed)\n",
    "\n",
    "# summarize encoded vector\n",
    "print('shape: ', vector.shape)\n",
    "print('vectors: ', vector.toarray())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7078d4b-aeb0-4461-885f-827e681a02c4",
   "metadata": {},
   "source": [
    "## Word Frequencies with TFIDF Vectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "3cad5232-0bb2-46cf-bc6b-9c31ef03da3b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'numpy.ndarray'>\n",
      "idfs:  [7.30991828 7.30991828 6.90445317 6.90445317 6.90445317 6.6167711\n",
      " 7.30991828 7.30991828 5.1126937  7.30991828 5.36400813 7.30991828\n",
      " 6.6167711  7.30991828 6.90445317 7.30991828 5.80584088 7.30991828\n",
      " 6.6167711  7.30991828 7.30991828 6.39362755 7.30991828 6.90445317\n",
      " 7.30991828 6.39362755 7.30991828 7.30991828 7.30991828 7.30991828\n",
      " 6.90445317 7.30991828 7.30991828 6.6167711  6.90445317 6.39362755\n",
      " 7.30991828 7.30991828 6.21130599 6.90445317 6.6167711  7.30991828\n",
      " 7.30991828 6.90445317 7.30991828 6.90445317 6.6167711  7.30991828\n",
      " 7.30991828 6.6167711  6.90445317 7.30991828 6.21130599 6.90445317\n",
      " 6.39362755 7.30991828 6.6167711  7.30991828 7.30991828 6.90445317\n",
      " 7.30991828 6.90445317 7.30991828 7.30991828 7.30991828 7.30991828\n",
      " 7.30991828 7.30991828 5.92362392 7.30991828 7.30991828 7.30991828\n",
      " 6.39362755 6.90445317 6.90445317 5.92362392 6.90445317 6.90445317\n",
      " 7.30991828 7.30991828 7.30991828 7.30991828 6.39362755 7.30991828\n",
      " 7.30991828 7.30991828 6.90445317 5.60517019 7.30991828 7.30991828\n",
      " 7.30991828 7.30991828 6.90445317 7.30991828 7.30991828 7.30991828\n",
      " 7.30991828 7.30991828 7.30991828 7.30991828 6.6167711  5.92362392\n",
      " 6.90445317 6.6167711  7.30991828 6.6167711  6.90445317 7.30991828\n",
      " 7.30991828 6.21130599 6.90445317 7.30991828 7.30991828 7.30991828\n",
      " 6.90445317 5.29501526 7.30991828 6.90445317 7.30991828 6.6167711\n",
      " 6.6167711  7.30991828 6.90445317 7.30991828 7.30991828 7.30991828\n",
      " 7.30991828 7.30991828 6.39362755 7.30991828 6.90445317 6.05715531\n",
      " 7.30991828 7.30991828 6.05715531 7.30991828 6.90445317 7.30991828\n",
      " 5.92362392 6.90445317 6.05715531 7.30991828 6.90445317 6.90445317\n",
      " 7.30991828 6.90445317 7.30991828 6.90445317 7.30991828 7.30991828\n",
      " 7.30991828 7.30991828 6.21130599 7.30991828 6.90445317 7.30991828\n",
      " 7.30991828 7.30991828 7.30991828 6.90445317 6.6167711  4.60186808\n",
      " 6.90445317 7.30991828 7.30991828 5.80584088 6.21130599 7.30991828\n",
      " 6.90445317 7.30991828 7.30991828 7.30991828 7.30991828 6.90445317\n",
      " 6.90445317 6.90445317 6.6167711  6.90445317 6.90445317 7.30991828\n",
      " 7.30991828 7.30991828 7.30991828 5.51815881 6.6167711  7.30991828\n",
      " 7.30991828 6.90445317 7.30991828 7.30991828 7.30991828 7.30991828\n",
      " 7.30991828 6.05715531 6.6167711  7.30991828 7.30991828 7.30991828\n",
      " 7.30991828 7.30991828 7.30991828 6.90445317 6.90445317 7.30991828\n",
      " 7.30991828 6.39362755 6.6167711  6.39362755 6.90445317 7.30991828\n",
      " 6.6167711  7.30991828 7.30991828 7.30991828 7.30991828 6.39362755\n",
      " 7.30991828 7.30991828 7.30991828 6.6167711  5.70048037 7.30991828\n",
      " 7.30991828 7.30991828 7.30991828 6.6167711  7.30991828 6.90445317\n",
      " 7.30991828 6.39362755 6.39362755 5.92362392 7.30991828 7.30991828\n",
      " 6.90445317 5.51815881 7.30991828 7.30991828 6.6167711  6.90445317\n",
      " 7.30991828 6.39362755 6.90445317 7.30991828 5.92362392 7.30991828\n",
      " 7.30991828 6.90445317 7.30991828 5.92362392 6.21130599 5.80584088\n",
      " 6.90445317 7.30991828 7.30991828 6.21130599 6.6167711  7.30991828\n",
      " 5.92362392 7.30991828 5.4381161  7.30991828 6.21130599 7.30991828\n",
      " 7.30991828 7.30991828 5.70048037 5.80584088 7.30991828 7.30991828\n",
      " 6.6167711  7.30991828 7.30991828 6.90445317 6.05715531 7.30991828\n",
      " 7.30991828 7.30991828 6.90445317 5.80584088 6.21130599 6.21130599\n",
      " 6.6167711  7.30991828 7.30991828 7.30991828 6.90445317 6.90445317\n",
      " 6.05715531 6.21130599 6.90445317 7.30991828 6.90445317 5.70048037\n",
      " 7.30991828 6.90445317 7.30991828 6.90445317 7.30991828 7.30991828\n",
      " 6.90445317 7.30991828 7.30991828 6.6167711  7.30991828 7.30991828\n",
      " 7.30991828 7.30991828 7.30991828 6.39362755 7.30991828 7.30991828\n",
      " 7.30991828 7.30991828 7.30991828 6.6167711  7.30991828 6.6167711\n",
      " 7.30991828 6.90445317 7.30991828 6.90445317 7.30991828 6.90445317\n",
      " 6.39362755 6.90445317 7.30991828 6.90445317 7.30991828 5.92362392\n",
      " 6.90445317 6.6167711  7.30991828 7.30991828 6.6167711  6.39362755\n",
      " 6.05715531 7.30991828 6.6167711  6.90445317 6.21130599 7.30991828\n",
      " 7.30991828 6.90445317 7.30991828 6.90445317 6.39362755 7.30991828\n",
      " 5.80584088 7.30991828 7.30991828 6.39362755 7.30991828 6.90445317\n",
      " 7.30991828 7.30991828 7.30991828 6.90445317 6.6167711  7.30991828\n",
      " 6.21130599 7.30991828 6.39362755 6.21130599 6.21130599 6.90445317\n",
      " 7.30991828 7.30991828 7.30991828 6.6167711  7.30991828 7.30991828\n",
      " 5.92362392 6.6167711  7.30991828 6.90445317 7.30991828 6.90445317\n",
      " 7.30991828 7.30991828 7.30991828 6.90445317 5.80584088 6.90445317\n",
      " 7.30991828 7.30991828 7.30991828 6.21130599 6.90445317 6.90445317\n",
      " 6.6167711  7.30991828 7.30991828 7.30991828 7.30991828 7.30991828\n",
      " 7.30991828 6.90445317 6.6167711  7.30991828 6.90445317 7.30991828\n",
      " 6.90445317 5.70048037 6.39362755 7.30991828 6.39362755 7.30991828\n",
      " 7.30991828 7.30991828 7.30991828 7.30991828 6.90445317 5.23047674\n",
      " 7.30991828 7.30991828 7.30991828 7.30991828 7.30991828 6.39362755\n",
      " 6.39362755 5.92362392 7.30991828 6.90445317 7.30991828 5.23047674\n",
      " 7.30991828 6.90445317 7.30991828 6.05715531 6.6167711  6.90445317\n",
      " 6.21130599 5.4381161  5.70048037 5.80584088 7.30991828 7.30991828\n",
      " 6.90445317 6.90445317 7.30991828 7.30991828 6.90445317 7.30991828\n",
      " 6.90445317 7.30991828 7.30991828 7.30991828]\n",
      "vectors:  [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n",
      "  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# create the transform\n",
    "vectorizer = TfidfVectorizer()\n",
    "\n",
    "# tokenize and build vocab\n",
    "vectorizer.fit(stemmed)\n",
    "\n",
    "# summarize\n",
    "print(type(vectorizer.idf_))\n",
    "print('idfs: ', vectorizer.idf_)\n",
    "\n",
    "# encode document\n",
    "vector = vectorizer.transform([stemmed[0]])\n",
    "\n",
    "# summarize encoded vector\n",
    "print('vectors: ', vector.toarray())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "f0407c57-d8d9-4078-a863-caecb935a962",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: >"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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8N4RAACCw3zdzvZLGE/FSFNTk0488cS05yZOnKgpU6Ykn7/00kv1/ve/P/mekhtvvFFnnnmm5syZo56eHt1yyy3atWuXvvKVr7jut6+vT5J47wgAAAHU19enSCRiu73gX5S3e/duVVW9977Yw4cP66tf/ar279+vyZMn65RTTtFzzz2nuXPnuj7mzJkztWfPHjU1NRX0HiXme1H27NnDe1FKBGNSehiT0sOYlB7GxJphGOrr69PMmTMd23l6A2u5cfvGGhQPY1J6GJPSw5iUHsYkP3w3DQAA8BXFCAAA8FVFFyN1dXX63ve+x8eISwhjUnoYk9LDmJQexiQ/Ff2eEQAA4L+K/p8RAADgP4oRAADgK4oRAADgK4oRAADgq4ouRu644w594AMfUH19vc444wz96U9/8juksrVhwwZ95jOf0cyZMxUKhfTQQw+lbTcMQ9/97nc1Y8YMNTQ0aPHixerq6kpr884772jZsmUKh8Nqbm7WFVdcof7+/iKeRflYtWqVTjvtNDU1Nam1tVUXX3yxtm/fntZmaGhIy5cv15QpUzRp0iQtXbpUBw4cSGuze/duXXjhhWpsbFRra6u+/e1va2RkpJinUjbuvPNOzZ8/X+FwWOFwWAsWLNDvf//75HbGw1+rV69WKBTSNddck3yOMSmcii1G/uu//kv/+I//qO9973t6+eWXddJJJ2nJkiU6ePCg36GVpYGBAZ100km64447LLf/4Ac/0G233aa77rpLL7zwgiZOnKglS5ZoaGgo2WbZsmV67bXXtG7dOj3yyCPasGGDvva1rxXrFMrK008/reXLl+v555/XunXrNDw8rE984hMaGBhItrn22mv1P//zP7r//vv19NNPa+/evfrc5z6X3D46OqoLL7xQ8Xhczz33nH7+859r7dq1+u53v+vHKQXeMccco9WrV2vz5s166aWXdP755+uiiy7Sa6+9Jonx8NOLL76on/70p5o/f37a84xJARkV6vTTTzeWL1+efDw6OmrMnDnTWLVqlY9RVQZJxoMPPph8nEgkjOnTpxu33HJL8rmenh6jrq7O+NWvfmUYhmG8/vrrhiTjxRdfTLb5/e9/b4RCIePtt98uWuzl6uDBg4Yk4+mnnzYM42j+a2pqjPvvvz/Z5v/+7/8MScamTZsMwzCMRx991KiqqjL279+fbHPnnXca4XDYiMVixT2BMjV58mTjZz/7GePho76+PqO9vd1Yt26d8bGPfcy4+uqrDcNgjRRaRf7PSDwe1+bNm7V48eLkc1VVVVq8eLE2bdrkY2SVaefOndq/f3/aeEQiEZ1xxhnJ8di0aZOam5t16qmnJtssXrxYVVVVeuGFF4oec7np7e2VJL3vfe+TJG3evFnDw8NpY3L88cfr2GOPTRuTefPmadq0ack2S5YsUTQaTf5rHrkZHR3Vfffdp4GBAS1YsIDx8NHy5ct14YUXpuVeYo0UWsG/tTcIuru7NTo6mjZBJGnatGl64403fIqqcu3fv1+SLMfD3LZ//361tramba+urtb73ve+ZBvkJpFI6JprrtHChQt14oknSjqa79raWjU3N6e1zRwTqzEzt8G7V199VQsWLNDQ0JAmTZqkBx98UHPnztUrr7zCePjgvvvu08svv6wXX3xxzDbWSGFVZDEC4D3Lly/Xtm3btHHjRr9DqXgf/vCH9corr6i3t1e/+c1v9KUvfUlPP/2032FVpD179ujqq6/WunXrVF9f73c4Za8if03T0tKiCRMmjHnX84EDBzR9+nSfoqpcZs6dxmP69Olj3lw8MjKid955hzHLw4oVK/TII4/oySef1DHHHJN8fvr06YrH4+rp6UlrnzkmVmNmboN3tbW1mjNnjk455RStWrVKJ510kn784x8zHj7YvHmzDh48qJNPPlnV1dWqrq7W008/rdtuu03V1dWaNm0aY1JAFVmM1NbW6pRTTtETTzyRfC6RSOiJJ57QggULfIysMs2ePVvTp09PG49oNKoXXnghOR4LFixQT0+PNm/enGyzfv16JRIJnXHGGUWPOegMw9CKFSv04IMPav369Zo9e3ba9lNOOUU1NTVpY7J9+3bt3r07bUxeffXVtCJx3bp1CofDmjt3bnFOpMwlEgnFYjHGwwcXXHCBXn31Vb3yyivJn1NPPVXLli1L/p0xKSC/30Hrl/vuu8+oq6sz1q5da7z++uvG1772NaO5uTntXc8onL6+PmPLli3Gli1bDEnGj370I2PLli3Grl27DMMwjNWrVxvNzc3Gww8/bGzdutW46KKLjNmzZxtHjhxJHuOTn/yk0dHRYbzwwgvGxo0bjfb2dqOzs9OvUwq0K6+80ohEIsZTTz1l7Nu3L/kzODiYbPP1r3/dOPbYY43169cbL730krFgwQJjwYIFye0jIyPGiSeeaHziE58wXnnlFeOxxx4zpk6daqxcudKPUwq866+/3nj66aeNnTt3Glu3bjWuv/56IxQKGX/84x8Nw2A8SkHqp2kMgzEppIotRgzDMG6//Xbj2GOPNWpra43TTz/deP755/0OqWw9+eSThqQxP1/60pcMwzj68d7vfOc7xrRp04y6ujrjggsuMLZv3552jEOHDhmdnZ3GpEmTjHA4bFx22WVGX1+fD2cTfFZjIcm49957k22OHDli/MM//IMxefJko7Gx0fjsZz9r7Nu3L+04f/3rX41PfepTRkNDg9HS0mJ885vfNIaHh4t8NuXh8ssvN2bNmmXU1tYaU6dONS644IJkIWIYjEcpyCxGGJPCCRmGYfjzfzIAAAAV+p4RAABQOihGAACAryhGAACAryhGAACAryhGAACAryhGAACAryhGAACAryhGAACAryhGAACAryhGAACAryhGAACAryhGAACAr/4/jmcGA98A1NYAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(vectorizer.idf_)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "81572e21-6651-4457-bc5b-e528de607d0a",
   "metadata": {},
   "source": [
    "## Working with a DataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "b3af3cd6-03d9-41cc-b707-8e5d7af4ef50",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "shape:  (1099, 448)\n"
     ]
    }
   ],
   "source": [
    "# Instantiate the CountVectorizer\n",
    "vectorizer = CountVectorizer()\n",
    "\n",
    "word_count_vector = vectorizer.fit_transform(stemmed)\n",
    "\n",
    "print('shape: ', word_count_vector.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "8c01e97e-2124-4170-8bc3-093087b3543e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>address</th>\n",
       "      <th>advic</th>\n",
       "      <th>advis</th>\n",
       "      <th>afraid</th>\n",
       "      <th>alert</th>\n",
       "      <th>alien</th>\n",
       "      <th>alik</th>\n",
       "      <th>ancient</th>\n",
       "      <th>anim</th>\n",
       "      <th>anoth</th>\n",
       "      <th>...</th>\n",
       "      <th>woman</th>\n",
       "      <th>womankind</th>\n",
       "      <th>wore</th>\n",
       "      <th>world</th>\n",
       "      <th>wrench</th>\n",
       "      <th>wrong</th>\n",
       "      <th>young</th>\n",
       "      <th>youth</th>\n",
       "      <th>zikrtt</th>\n",
       "      <th>zikru</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1094</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1095</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1096</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1097</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1098</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1099 rows × 448 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      address  advic  advis  afraid  alert  alien  alik  ancient  anim  anoth  \\\n",
       "0           0      0      0       0      0      0     0        0     0      0   \n",
       "1           0      0      0       0      0      0     0        0     0      0   \n",
       "2           0      0      0       0      0      0     0        0     0      0   \n",
       "3           0      0      0       0      0      0     0        0     0      0   \n",
       "4           0      0      0       0      0      0     0        0     0      0   \n",
       "...       ...    ...    ...     ...    ...    ...   ...      ...   ...    ...   \n",
       "1094        0      0      0       0      0      0     0        0     0      0   \n",
       "1095        0      0      0       0      0      0     0        0     0      0   \n",
       "1096        0      0      0       0      0      0     0        0     0      0   \n",
       "1097        0      0      0       0      0      0     0        0     0      0   \n",
       "1098        0      0      0       0      0      0     0        0     0      0   \n",
       "\n",
       "      ...  woman  womankind  wore  world  wrench  wrong  young  youth  zikrtt  \\\n",
       "0     ...      0          0     0      0       0      0      0      0       0   \n",
       "1     ...      0          0     0      0       0      0      0      0       0   \n",
       "2     ...      0          0     0      0       0      0      0      0       0   \n",
       "3     ...      0          0     0      0       0      0      0      0       0   \n",
       "4     ...      0          0     0      0       0      0      0      0       0   \n",
       "...   ...    ...        ...   ...    ...     ...    ...    ...    ...     ...   \n",
       "1094  ...      0          0     0      0       0      0      0      0       0   \n",
       "1095  ...      0          0     0      0       0      0      0      0       0   \n",
       "1096  ...      0          0     0      0       0      0      0      0       0   \n",
       "1097  ...      0          0     0      0       0      0      0      0       0   \n",
       "1098  ...      0          0     0      0       0      0      0      0       0   \n",
       "\n",
       "      zikru  \n",
       "0         0  \n",
       "1         0  \n",
       "2         0  \n",
       "3         0  \n",
       "4         0  \n",
       "...     ...  \n",
       "1094      0  \n",
       "1095      0  \n",
       "1096      0  \n",
       "1097      0  \n",
       "1098      0  \n",
       "\n",
       "[1099 rows x 448 columns]"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Create the Dataframe\n",
    "\n",
    "df = pd.DataFrame(word_count_vector.toarray(), columns = vectorizer.get_feature_names_out())\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bf6c5f4e-3a57-492f-8b7c-32fa1321518d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
