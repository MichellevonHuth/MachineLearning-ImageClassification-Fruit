{
 "cells": [
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "e721f1be-c60a-4f05-81cc-d0f7643ef14f",
   "metadata": {},
   "source": [
    "### Install dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cea5c26e-bfb2-4589-891a-f3e496d1cf4d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-14T17:41:02.983470Z",
     "iopub.status.busy": "2023-05-14T17:41:02.983289Z",
     "iopub.status.idle": "2023-05-14T17:41:06.350602Z",
     "shell.execute_reply": "2023-05-14T17:41:06.348875Z",
     "shell.execute_reply.started": "2023-05-14T17:41:02.983451Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install scikit-learn scikit-image prettytable seaborn --quiet"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "83b34040-c3f1-4776-818f-87c39f0f6a3e",
   "metadata": {},
   "source": [
    "### Import libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ba30cb1f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-10T22:33:44.896996Z",
     "start_time": "2023-05-10T22:33:42.811145Z"
    },
    "execution": {
     "iopub.execute_input": "2023-05-14T17:41:06.353847Z",
     "iopub.status.busy": "2023-05-14T17:41:06.353228Z",
     "iopub.status.idle": "2023-05-14T17:41:07.502094Z",
     "shell.execute_reply": "2023-05-14T17:41:07.501169Z",
     "shell.execute_reply.started": "2023-05-14T17:41:06.353781Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import PIL\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import random\n",
    "import pickle\n",
    "import seaborn as sns\n",
    "from PIL import Image\n",
    "import matplotlib.pyplot as plt\n",
    "from prettytable import PrettyTable\n",
    "\n",
    "from skimage.transform import resize\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict\n",
    "from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict, learning_curve\n",
    "from sklearn.metrics import roc_curve, auc\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support\n",
    "from sklearn.dummy import DummyClassifier"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "24b669f5-4dec-4481-89bd-5abde9bed804",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T10:07:30.710777Z",
     "iopub.status.busy": "2023-05-11T10:07:30.709975Z",
     "iopub.status.idle": "2023-05-11T10:07:30.716582Z",
     "shell.execute_reply": "2023-05-11T10:07:30.715338Z",
     "shell.execute_reply.started": "2023-05-11T10:07:30.710709Z"
    },
    "tags": []
   },
   "source": [
    "### Load dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b4ab5b34",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-10T22:33:50.974557Z",
     "start_time": "2023-05-10T22:33:47.677744Z"
    },
    "execution": {
     "iopub.execute_input": "2023-05-14T17:41:07.503594Z",
     "iopub.status.busy": "2023-05-14T17:41:07.503243Z",
     "iopub.status.idle": "2023-05-14T17:41:09.510143Z",
     "shell.execute_reply": "2023-05-14T17:41:09.509235Z",
     "shell.execute_reply.started": "2023-05-14T17:41:07.503571Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "## Re-run EDA.ipynb if issues with loading pickle ##\n",
    "# Load the pickled dataset (12335 augmented images; 128x128 pixels; binary labels - Fresh/Rotten; 80:20 train/test split)\n",
    "X_train, X_test, y_train, y_test = pd.read_pickle(open(\"data_binary_128_split.pkl\", \"rb\"))"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "4ec3bde2-b3fd-4a03-ad7d-25e7a655d2a6",
   "metadata": {},
   "source": [
    "### Data preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a2026360-1a1e-43a1-8eef-98491ea15015",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-10T22:34:01.955202Z",
     "start_time": "2023-05-10T22:33:51.980986Z"
    },
    "execution": {
     "iopub.execute_input": "2023-05-14T17:41:09.513878Z",
     "iopub.status.busy": "2023-05-14T17:41:09.513435Z",
     "iopub.status.idle": "2023-05-14T17:41:14.742368Z",
     "shell.execute_reply": "2023-05-14T17:41:14.741191Z",
     "shell.execute_reply.started": "2023-05-14T17:41:09.513851Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Preprocess image data to flat 1D arrays before PCA and SVM\n",
    "# Flatten X_train\n",
    "X_train_flat = []\n",
    "for image in X_train:\n",
    "    image_flattened = image.flatten()\n",
    "    X_train_flat.append(image_flattened)\n",
    "X_train_flat = np.array(X_train_flat)\n",
    "\n",
    "# Flatten X_test\n",
    "X_test_flat = []\n",
    "for image in X_test:\n",
    "    image_flattened = image.flatten()\n",
    "    X_test_flat.append(image_flattened)\n",
    "X_test_flat = np.array(X_test_flat)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "6b414b82-42d5-4d3a-90fd-e6e0573f4f28",
   "metadata": {},
   "source": [
    "### Build and train the SVM classifier using grid search for hyperparameter tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "87a3fe85-1371-4255-a51a-ab103faa466e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-14T17:41:14.744034Z",
     "iopub.status.busy": "2023-05-14T17:41:14.743795Z",
     "iopub.status.idle": "2023-05-14T18:24:06.475645Z",
     "shell.execute_reply": "2023-05-14T18:24:06.474671Z",
     "shell.execute_reply.started": "2023-05-14T17:41:14.744013Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 7h 25min 47s, sys: 3h 3min 23s, total: 10h 29min 11s\n",
      "Wall time: 42min 51s\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=5,\n",
       "             estimator=Pipeline(steps=[(&#x27;scalar&#x27;, StandardScaler()),\n",
       "                                       (&#x27;pca&#x27;, PCA(random_state=42)),\n",
       "                                       (&#x27;svc&#x27;,\n",
       "                                        SVC(probability=True,\n",
       "                                            random_state=42))]),\n",
       "             param_grid={&#x27;pca__n_components&#x27;: [50, 75, 100],\n",
       "                         &#x27;svc__C&#x27;: [10, 40, 60, 100], &#x27;svc__gamma&#x27;: [&#x27;scale&#x27;],\n",
       "                         &#x27;svc__kernel&#x27;: [&#x27;rbf&#x27;]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=5,\n",
       "             estimator=Pipeline(steps=[(&#x27;scalar&#x27;, StandardScaler()),\n",
       "                                       (&#x27;pca&#x27;, PCA(random_state=42)),\n",
       "                                       (&#x27;svc&#x27;,\n",
       "                                        SVC(probability=True,\n",
       "                                            random_state=42))]),\n",
       "             param_grid={&#x27;pca__n_components&#x27;: [50, 75, 100],\n",
       "                         &#x27;svc__C&#x27;: [10, 40, 60, 100], &#x27;svc__gamma&#x27;: [&#x27;scale&#x27;],\n",
       "                         &#x27;svc__kernel&#x27;: [&#x27;rbf&#x27;]})</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: Pipeline</label><div class=\"sk-toggleable__content\"><pre>Pipeline(steps=[(&#x27;scalar&#x27;, StandardScaler()), (&#x27;pca&#x27;, PCA(random_state=42)),\n",
       "                (&#x27;svc&#x27;, SVC(probability=True, random_state=42))])</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">StandardScaler</label><div class=\"sk-toggleable__content\"><pre>StandardScaler()</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">PCA</label><div class=\"sk-toggleable__content\"><pre>PCA(random_state=42)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" ><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SVC</label><div class=\"sk-toggleable__content\"><pre>SVC(probability=True, random_state=42)</pre></div></div></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=5,\n",
       "             estimator=Pipeline(steps=[('scalar', StandardScaler()),\n",
       "                                       ('pca', PCA(random_state=42)),\n",
       "                                       ('svc',\n",
       "                                        SVC(probability=True,\n",
       "                                            random_state=42))]),\n",
       "             param_grid={'pca__n_components': [50, 75, 100],\n",
       "                         'svc__C': [10, 40, 60, 100], 'svc__gamma': ['scale'],\n",
       "                         'svc__kernel': ['rbf']})"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create the pipeline\n",
    "pipeline = Pipeline([\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('pca', PCA(random_state=42)),\n",
    "    ('svc', SVC(random_state=42, probability=True))\n",
    "])\n",
    "\n",
    "# Define the parameter grid for the pipeline - updated based on previous GridSearch results\n",
    "param_grid = {\n",
    "    'pca__n_components': [50, 75, 100],\n",
    "    'svc__C': [10, 40, 60, 100],\n",
    "    'svc__kernel': ['rbf'], \n",
    "    'svc__gamma': ['scale']\n",
    "}\n",
    "\n",
    "# Instatiate GridSearchCV object\n",
    "grid_search = GridSearchCV(pipeline, param_grid, cv=5)\n",
    "\n",
    "# Fit GridSearchCV object to the training data\n",
    "%time grid_search.fit(X_train_flat, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b332bcbc-fab3-4eac-a5e0-2fc1949376df",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-14T18:24:06.476894Z",
     "iopub.status.busy": "2023-05-14T18:24:06.476680Z",
     "iopub.status.idle": "2023-05-14T18:24:06.482439Z",
     "shell.execute_reply": "2023-05-14T18:24:06.481846Z",
     "shell.execute_reply.started": "2023-05-14T18:24:06.476872Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best hyperparameters: {'pca__n_components': 100, 'svc__C': 40, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'}\n",
      "Best score: 0.9544991632222442\n"
     ]
    }
   ],
   "source": [
    "# Print the best hyperparameters and the corresponding score\n",
    "print('Best hyperparameters:', grid_search.best_params_)\n",
    "print('Best score:', grid_search.best_score_)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "1da8c947-dc07-408f-92fe-918a524f4ef9",
   "metadata": {},
   "source": [
    "### Predict and evaluate performance on the train set (in-sample)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "11847744-07b5-47fb-8f04-46566a0f105f",
   "metadata": {},
   "source": [
    "#### Predict on the train set (5-fold cross validation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f0b932ec-474b-4928-804d-f41345bf1b7c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-14T18:24:06.483295Z",
     "iopub.status.busy": "2023-05-14T18:24:06.483127Z",
     "iopub.status.idle": "2023-05-14T18:31:44.269568Z",
     "shell.execute_reply": "2023-05-14T18:31:44.268317Z",
     "shell.execute_reply.started": "2023-05-14T18:24:06.483280Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Cross-val predict on train set\n",
    "y_train_pred = cross_val_predict(grid_search.best_estimator_, X_train_flat, y_train, cv=5, method = 'predict')\n",
    "y_train_pred_prob = cross_val_predict(grid_search.best_estimator_, X_train_flat, y_train, cv=5, method = 'predict_proba')"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "668d16fa-8ba7-402f-999b-b77889abe707",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T10:02:48.759539Z",
     "iopub.status.busy": "2023-05-11T10:02:48.758898Z",
     "iopub.status.idle": "2023-05-11T10:02:48.764262Z",
     "shell.execute_reply": "2023-05-11T10:02:48.763253Z",
     "shell.execute_reply.started": "2023-05-11T10:02:48.759496Z"
    },
    "tags": []
   },
   "source": [
    "#### Model evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b2550e37-fd17-4f99-8cd4-cef314666294",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-14T18:31:44.271263Z",
     "iopub.status.busy": "2023-05-14T18:31:44.271050Z",
     "iopub.status.idle": "2023-05-14T18:31:44.658466Z",
     "shell.execute_reply": "2023-05-14T18:31:44.657716Z",
     "shell.execute_reply.started": "2023-05-14T18:31:44.271242Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AUC for fold 1: 0.99\n",
      "AUC for fold 2: 0.99\n",
      "AUC for fold 3: 0.99\n",
      "AUC for fold 4: 0.99\n",
      "AUC for fold 5: 0.99\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 600x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "## Adapted from example in Scikit-Learn documentation - https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html\n",
    "# Initialize a StratifiedKFold object\n",
    "cv = StratifiedKFold(n_splits=5)\n",
    "y_train_binary = (y_train == 'Rotten').astype(int)\n",
    "\n",
    "# Define the True Positive Rate (TPR), Area Under Curve (AUC) lists\n",
    "tprs = []\n",
    "aucs = []\n",
    "mean_fpr = np.linspace(0, 1, 100)\n",
    "\n",
    "# Plotting\n",
    "fig, ax = plt.subplots(figsize=(6, 6))\n",
    "\n",
    "# Loop over each fold\n",
    "for i, (train, test) in enumerate(cv.split(X_train_flat, y_train_binary)):\n",
    "\n",
    "    # Compute ROC curve and ROC area for each fold using the `roc_curve` function\n",
    "    fpr, tpr, _ = roc_curve(y_train_binary[test], y_train_pred_prob[test, 1])\n",
    "    roc_auc = auc(fpr, tpr)\n",
    "\n",
    "    # Interpolate the TPR\n",
    "    interp_tpr = np.interp(mean_fpr, fpr, tpr)\n",
    "    interp_tpr[0] = 0.0\n",
    "    tprs.append(interp_tpr)\n",
    "    aucs.append(roc_auc)\n",
    "    \n",
    "    # Print AUC for this fold\n",
    "    print(f\"AUC for fold {i+1}: {roc_auc:.2f}\")\n",
    "\n",
    "    # Plot ROC for this fold\n",
    "    ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f\"ROC fold {i} (AUC = {roc_auc:.2f})\")\n",
    "\n",
    "ax.plot([0, 1], [0, 1], \"k--\", label=\"Chance level (AUC = 0.5)\")\n",
    "\n",
    "# Compute the mean ROC curve\n",
    "mean_tpr = np.mean(tprs, axis=0)\n",
    "mean_tpr[-1] = 1.0\n",
    "mean_auc = auc(mean_fpr, mean_tpr)\n",
    "std_auc = np.std(aucs)\n",
    "\n",
    "# Plot the mean ROC curve\n",
    "ax.plot(mean_fpr, mean_tpr, color=\"b\", label=r\"Mean ROC (AUC = %0.2f $\\pm$ %0.2f)\" % (mean_auc, std_auc), lw=2, alpha=0.8)\n",
    "\n",
    "# Plot the standard deviation around the mean ROC curve\n",
    "std_tpr = np.std(tprs, axis=0)\n",
    "tprs_upper = np.minimum(mean_tpr + std_tpr, 1)\n",
    "tprs_lower = np.maximum(mean_tpr - std_tpr, 0)\n",
    "ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=\"grey\", alpha=0.2, label=r\"$\\pm$ 1 std. dev.\")\n",
    "\n",
    "# Set plot properties\n",
    "ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], xlabel=\"False Positive Rate\", ylabel=\"True Positive Rate\", title=\"SVM\\n ROC Curves for Each Fold\")\n",
    "ax.legend(loc=\"lower right\")\n",
    "\n",
    "# Show the plot\n",
    "plt.show()"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "336c7dcc-c59b-4f26-9e89-98549742321d",
   "metadata": {},
   "source": [
    "### Predict and evaluate performance on test set (out-of-sample)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "fe10ebe3-8b42-4dcc-ad38-1445a5a15452",
   "metadata": {},
   "source": [
    "#### Predict on test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "47ac8dec-b429-4019-a3df-6917c17e89b7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-14T18:31:44.659437Z",
     "iopub.status.busy": "2023-05-14T18:31:44.659255Z",
     "iopub.status.idle": "2023-05-14T18:31:46.805746Z",
     "shell.execute_reply": "2023-05-14T18:31:46.804739Z",
     "shell.execute_reply.started": "2023-05-14T18:31:44.659420Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Predict on test\n",
    "y_test_pred = grid_search.best_estimator_.predict(X_test_flat)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "efe2734e-8b5e-4a87-8f20-fa255ad28e53",
   "metadata": {},
   "source": [
    "#### Confusion matrix (SVM, test data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "502ce2a4-ed3b-453c-ad76-fc785b817031",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-14T18:31:46.806764Z",
     "iopub.status.busy": "2023-05-14T18:31:46.806583Z",
     "iopub.status.idle": "2023-05-14T18:31:46.928343Z",
     "shell.execute_reply": "2023-05-14T18:31:46.927654Z",
     "shell.execute_reply.started": "2023-05-14T18:31:46.806748Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Confusion Matrix (SVM, Test Data): \n",
      "+----------+-------------+-------------+\n",
      "|          | Predicted 0 | Predicted 1 |\n",
      "+----------+-------------+-------------+\n",
      "| Actual 0 |     1205    |      51     |\n",
      "| Actual 1 |      47     |     1164    |\n",
      "+----------+-------------+-------------+\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 300x250 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Confusion matrix\n",
    "data = confusion_matrix(y_test, y_test_pred)\n",
    "tab2 = PrettyTable()\n",
    "tab2.field_names = [\"\", \"Predicted 0\", \"Predicted 1\"]\n",
    "tab2.add_row([\"Actual 0\", data[0][0], data[0][1]])\n",
    "tab2.add_row([\"Actual 1\", data[1][0], data[1][1]])\n",
    "\n",
    "print(f\"\\nConfusion Matrix (SVM, Test Data): \\n{tab2}\")\n",
    "\n",
    "# Plot the confusion matrix using seaborn\n",
    "plt.figure(figsize=(3,2.5))\n",
    "sns.heatmap(data, annot=True, cmap='Blues', fmt='g')\n",
    "plt.xlabel('Predicted label')\n",
    "plt.ylabel('True label')\n",
    "plt.show()"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "c93730be-bca4-4fdd-b644-990dd702e7dc",
   "metadata": {},
   "source": [
    "#### Performance Measures (SVM, test data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "b1ebd582-b6c5-4968-8887-09c5abcf8945",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-14T18:31:48.098177Z",
     "iopub.status.busy": "2023-05-14T18:31:48.098011Z",
     "iopub.status.idle": "2023-05-14T18:31:48.112716Z",
     "shell.execute_reply": "2023-05-14T18:31:48.112227Z",
     "shell.execute_reply.started": "2023-05-14T18:31:48.098162Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Performance Measure Table (SVM, Test Data): \n",
      "+---------------------+----------------+-----------------+\n",
      "| Performance Measure | Label 0: Fresh | Label 1: Rotten |\n",
      "+---------------------+----------------+-----------------+\n",
      "|      Precision      |     0.9625     |      0.958      |\n",
      "|        Recall       |     0.9594     |      0.9612     |\n",
      "|          F1         |     0.9609     |      0.9596     |\n",
      "|       Support       |      1256      |       1211      |\n",
      "+---------------------+----------------+-----------------+\n",
      "\n",
      "SVM Overall Accuracy Score (Test data): 0.9603\n"
     ]
    }
   ],
   "source": [
    "# Performance scores\n",
    "y_test_pred_scores = precision_recall_fscore_support(y_test, y_test_pred)\n",
    "\n",
    "# Create performance measure table\n",
    "tab3 = PrettyTable([\"Performance Measure\", \"Label 0: Fresh\", \"Label 1: Rotten\"])\n",
    "measure_names = [\"Precision\", \"Recall\", \"F1\", \"Support\"]\n",
    "\n",
    "# Add rows to table\n",
    "tab3.add_row([measure_names[0]] + y_test_pred_scores[0].round(4).tolist())\n",
    "tab3.add_row([measure_names[1]] + y_test_pred_scores[1].round(4).tolist())\n",
    "tab3.add_row([measure_names[2]] + y_test_pred_scores[2].round(4).tolist())\n",
    "tab3.add_row([measure_names[3]] + y_test_pred_scores[3].round(4).tolist())\n",
    "\n",
    "# Print performance measure table and accuracy\n",
    "print(f\"\\nPerformance Measure Table (SVM, Test Data): \\n{tab3}\")\n",
    "print(f\"\\nSVM Overall Accuracy Score (Test data): {accuracy_score(y_test, y_test_pred).round(4)}\")"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "42fa54d8-ff4f-4dfa-9b83-1cc7794cb6d3",
   "metadata": {},
   "source": [
    "-----"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "0ac04775-b623-4763-a9be-028abc33a511",
   "metadata": {},
   "source": [
    "### Complexity analysis - Baseline Model Comparison\n",
    "- Baseline 1 = all default SVM hyperparameters\n",
    "- Baseline 2 = PCA first, then all default SVM hyperparameters\n",
    "- Tuned model = optimal hyperparameters found from grid searches"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "12cba458-033a-46e2-b1ce-34a666f5fc48",
   "metadata": {},
   "source": [
    "#### Baseline 1 (default SVM params, no PCA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4361890a-0427-4127-a492-e2668e86ec49",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-14T17:42:31.000708Z",
     "iopub.status.busy": "2023-05-14T17:42:30.999817Z",
     "iopub.status.idle": "2023-05-14T20:07:36.790492Z",
     "shell.execute_reply": "2023-05-14T20:07:36.786932Z",
     "shell.execute_reply.started": "2023-05-14T17:42:31.000650Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 10h 13min 33s, sys: 1d 4h 10min 18s, total: 1d 14h 23min 52s\n",
      "Wall time: 1h 16min 57s\n",
      "\n",
      "Performance Measure Table (SVM Baseline 1 Model (no PCA), Test Data): \n",
      "+---------------------+----------------+-----------------+\n",
      "| Performance Measure | Label 0: fresh | Label 1: rotten |\n",
      "+---------------------+----------------+-----------------+\n",
      "|      Precision      |     0.9174     |      0.9247     |\n",
      "|        Recall       |     0.9283     |      0.9133     |\n",
      "|          F1         |     0.9228     |      0.919      |\n",
      "|       Support       |      1256      |       1211      |\n",
      "+---------------------+----------------+-----------------+\n",
      "\n",
      "SVM Baseline 1 Model Accuracy Score: 0.921\n"
     ]
    }
   ],
   "source": [
    "# Baseline model with all default SVM hyperparameters #17:40\n",
    "\n",
    "# Instatiate baseline1 model\n",
    "baseline_1 = Pipeline([\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('svc', SVC(random_state=42)) # empty SVM -> use default params\n",
    "]) \n",
    "\n",
    "# Fit the baseline model to the training data\n",
    "%time baseline_1.fit(X_train_flat, y_train)\n",
    "\n",
    "# Evaluate baseline model\n",
    "y_test_pred_base = baseline_1.predict(X_test_flat)\n",
    "base_scores_test = precision_recall_fscore_support(y_test, y_test_pred_base)\n",
    "\n",
    "# Create performance measure table for the baseline model\n",
    "measure_names = [\"Precision\", \"Recall\", \"F1\", \"Support\"]\n",
    "tab4 = PrettyTable([\"Performance Measure\", \"Label 0: fresh\", \"Label 1: rotten\"])\n",
    "tab4.add_row([measure_names[0]] + base_scores_test[0].round(4).tolist())\n",
    "tab4.add_row([measure_names[1]] + base_scores_test[1].round(4).tolist())\n",
    "tab4.add_row([measure_names[2]] + base_scores_test[2].round(4).tolist())\n",
    "tab4.add_row([measure_names[3]] + base_scores_test[3].round(4).tolist())\n",
    "\n",
    "# Print performance measure table and accuracy for both models\n",
    "print(f\"\\nPerformance Measure Table (SVM Baseline 1 Model (no PCA), Test Data): \\n{tab4}\")\n",
    "print(f\"\\nSVM Baseline 1 Model Accuracy Score: {accuracy_score(y_test, y_test_pred_base).round(4)}\")"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "dde3fab6-a83d-4abc-ae92-1008c086e15b",
   "metadata": {
    "tags": []
   },
   "source": [
    "#### Baseline 2 (default SVM params, with PCA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "c9c2a8b8-9c31-454f-86c0-56b3621ceb9d",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-11T07:32:36.190373Z",
     "start_time": "2023-05-11T07:32:31.104011Z"
    },
    "execution": {
     "iopub.execute_input": "2023-05-14T17:44:17.677608Z",
     "iopub.status.busy": "2023-05-14T17:44:17.676698Z",
     "iopub.status.idle": "2023-05-14T17:45:00.341762Z",
     "shell.execute_reply": "2023-05-14T17:45:00.339222Z",
     "shell.execute_reply.started": "2023-05-14T17:44:17.677552Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 9min 31s, sys: 3min 56s, total: 13min 28s\n",
      "Wall time: 40.1 s\n",
      "\n",
      "Performance Measure Table (SVM Baseline 2 Model (with PCA), Test Data): \n",
      "+---------------------+----------------+-----------------+\n",
      "| Performance Measure | Label 0: fresh | Label 1: rotten |\n",
      "+---------------------+----------------+-----------------+\n",
      "|      Precision      |     0.9175     |      0.9263     |\n",
      "|        Recall       |     0.9299     |      0.9133     |\n",
      "|          F1         |     0.9237     |      0.9198     |\n",
      "|       Support       |      1256      |       1211      |\n",
      "+---------------------+----------------+-----------------+\n",
      "\n",
      "SVM Baseline 2 Model Accuracy Score: 0.9218\n"
     ]
    }
   ],
   "source": [
    "# Baseline model with all default SVM hyperparameters + PCA\n",
    "\n",
    "# Instatiate baseline2 model\n",
    "baseline_2 = Pipeline([\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('pca', PCA(n_components=100, random_state=42)),\n",
    "    ('svc', SVC(random_state=42)) # empty SVM -> use default params\n",
    "]) \n",
    "\n",
    "# Fit the baseline2 model to the training data\n",
    "%time baseline_2.fit(X_train_flat, y_train)\n",
    "\n",
    "# Evaluate baseline2 model\n",
    "y_test_pred_base2 = baseline_2.predict(X_test_flat)\n",
    "base2_scores_test = precision_recall_fscore_support(y_test, y_test_pred_base2)\n",
    "\n",
    "# Create performance measure table for the baseline model\n",
    "measure_names = [\"Precision\", \"Recall\", \"F1\", \"Support\"]\n",
    "tab6 = PrettyTable([\"Performance Measure\", \"Label 0: fresh\", \"Label 1: rotten\"])\n",
    "tab6.add_row([measure_names[0]] + base2_scores_test[0].round(4).tolist())\n",
    "tab6.add_row([measure_names[1]] + base2_scores_test[1].round(4).tolist())\n",
    "tab6.add_row([measure_names[2]] + base2_scores_test[2].round(4).tolist())\n",
    "tab6.add_row([measure_names[3]] + base2_scores_test[3].round(4).tolist())\n",
    "\n",
    "# Print performance measure table and accuracy for both models\n",
    "print(f\"\\nPerformance Measure Table (SVM Baseline 2 Model (with PCA), Test Data): \\n{tab6}\")\n",
    "print(f\"\\nSVM Baseline 2 Model Accuracy Score: {accuracy_score(y_test, y_test_pred_base2).round(4)}\")"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "674e0686-bb45-409a-a077-2d6ba3abb780",
   "metadata": {},
   "source": [
    "#### Tuned (optimal SVM params, with PCA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "efba1802",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-11T08:54:14.775772Z",
     "start_time": "2023-05-11T08:54:14.749576Z"
    },
    "execution": {
     "iopub.execute_input": "2023-05-14T17:44:08.041889Z",
     "iopub.status.busy": "2023-05-14T17:44:08.041060Z",
     "iopub.status.idle": "2023-05-14T17:44:57.975764Z",
     "shell.execute_reply": "2023-05-14T17:44:57.974736Z",
     "shell.execute_reply.started": "2023-05-14T17:44:08.041815Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 9min 13s, sys: 7min 16s, total: 16min 29s\n",
      "Wall time: 47.3 s\n",
      "\n",
      "Performance Measure Table (Tuned SVM Model, Test Data): \n",
      "+---------------------+----------------+-----------------+\n",
      "| Performance Measure | Label 0: fresh | Label 1: rotten |\n",
      "+---------------------+----------------+-----------------+\n",
      "|      Precision      |     0.9625     |      0.958      |\n",
      "|        Recall       |     0.9594     |      0.9612     |\n",
      "|          F1         |     0.9609     |      0.9596     |\n",
      "|       Support       |      1256      |       1211      |\n",
      "+---------------------+----------------+-----------------+\n",
      "\n",
      "Tuned SVM Model Accuracy Score: 0.9603\n"
     ]
    }
   ],
   "source": [
    "# Instatiate tuned model\n",
    "tuned_model = Pipeline([\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('pca', PCA(n_components=100, random_state=42)),\n",
    "    ('svc', SVC(C=40, kernel='rbf', gamma='scale', random_state=42)) # Optimal SVM params from gridsearch\n",
    "])\n",
    "\n",
    "# Fit the tuned model to the training data\n",
    "%time tuned_model.fit(X_train_flat, y_train)\n",
    "\n",
    "# Evaluate tuned model\n",
    "y_test_pred_tuned = tuned_model.predict(X_test_flat)\n",
    "tuned_scores_test = precision_recall_fscore_support(y_test, y_test_pred_tuned)\n",
    "\n",
    "# Create performance measure table for the tuned model\n",
    "measure_names = [\"Precision\", \"Recall\", \"F1\", \"Support\"]\n",
    "tab5 = PrettyTable([\"Performance Measure\", \"Label 0: fresh\", \"Label 1: rotten\"])\n",
    "tab5.add_row([measure_names[0]] + tuned_scores_test[0].round(4).tolist())\n",
    "tab5.add_row([measure_names[1]] + tuned_scores_test[1].round(4).tolist())\n",
    "tab5.add_row([measure_names[2]] + tuned_scores_test[2].round(4).tolist())\n",
    "tab5.add_row([measure_names[3]] + tuned_scores_test[3].round(4).tolist())\n",
    "\n",
    "# Print performance measure table and accuracy for both models\n",
    "print(f\"\\nPerformance Measure Table (Tuned SVM Model, Test Data): \\n{tab5}\")\n",
    "print(f\"\\nTuned SVM Model Accuracy Score: {accuracy_score(y_test, y_test_pred_tuned).round(4)}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 ",
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
   "version": "3.10.10"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": false,
   "skip_h1_title": true,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  },
  "toc-autonumbering": false,
  "toc-showcode": false,
  "toc-showmarkdowntxt": false,
  "toc-showtags": false
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
