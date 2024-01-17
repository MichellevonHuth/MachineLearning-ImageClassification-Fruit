{
 "cells": [
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "c74f2584-603d-41ed-8af4-4069ab079549",
   "metadata": {},
   "source": [
    "### Install dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "baa336f6-3870-47c0-855c-0077200a02c0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T13:37:23.057581Z",
     "iopub.status.busy": "2023-05-11T13:37:23.057345Z",
     "iopub.status.idle": "2023-05-11T13:37:26.860276Z",
     "shell.execute_reply": "2023-05-11T13:37:26.856621Z",
     "shell.execute_reply.started": "2023-05-11T13:37:23.057556Z"
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
   "id": "ca87ff1d-2dcc-4be7-abca-b3f5f3cc35ba",
   "metadata": {},
   "source": [
    "### Import libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "cdd69e26-0905-4e1c-854c-422835f78c90",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T13:37:26.866236Z",
     "iopub.status.busy": "2023-05-11T13:37:26.865575Z",
     "iopub.status.idle": "2023-05-11T13:37:28.615359Z",
     "shell.execute_reply": "2023-05-11T13:37:28.614441Z",
     "shell.execute_reply.started": "2023-05-11T13:37:26.866188Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import PIL\n",
    "import numpy as np\n",
    "import random\n",
    "import pickle\n",
    "import seaborn as sns\n",
    "import os\n",
    "import pandas as pd\n",
    "from PIL import Image\n",
    "import matplotlib.pyplot as plt\n",
    "from prettytable import PrettyTable\n",
    "from sklearn.metrics import roc_curve, auc\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "\n",
    "from skimage.transform import resize\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict\n",
    "from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict, learning_curve\n",
    "from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support\n",
    "from sklearn.dummy import DummyClassifier"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "fe3c76fe-81d9-4eee-bb17-bbee84d2047d",
   "metadata": {},
   "source": [
    "### Load dataset "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fca95aa8-3b09-4aa5-b218-720b923cdcd2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T13:37:28.616690Z",
     "iopub.status.busy": "2023-05-11T13:37:28.616350Z",
     "iopub.status.idle": "2023-05-11T13:37:30.607027Z",
     "shell.execute_reply": "2023-05-11T13:37:30.606098Z",
     "shell.execute_reply.started": "2023-05-11T13:37:28.616666Z"
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
   "id": "4f0458b0-371d-40eb-951c-4a031cfffe8f",
   "metadata": {},
   "source": [
    "### Data preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "17e9b793-0244-49db-9c99-ada9dd00e780",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T13:37:30.608492Z",
     "iopub.status.busy": "2023-05-11T13:37:30.608270Z",
     "iopub.status.idle": "2023-05-11T13:37:35.492283Z",
     "shell.execute_reply": "2023-05-11T13:37:35.491433Z",
     "shell.execute_reply.started": "2023-05-11T13:37:30.608473Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Flatten images into 1D arrays\n",
    "X_train_flat = []\n",
    "for image in X_train:\n",
    "    image_flattened = image.flatten()\n",
    "    X_train_flat.append(image_flattened)\n",
    "X_train_flat = np.array(X_train_flat)\n",
    "\n",
    "# Repeat the same process for the test set\n",
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
   "id": "a2531376-8433-4022-abc9-1c8ec2ff8e91",
   "metadata": {},
   "source": [
    "### Build and train the Random Forest classifier using grid search for hyperparameter tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6af6414a-54e3-428a-bb5e-91ddcb5bd3c9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T13:37:35.493537Z",
     "iopub.status.busy": "2023-05-11T13:37:35.493312Z",
     "iopub.status.idle": "2023-05-11T13:46:31.804535Z",
     "shell.execute_reply": "2023-05-11T13:46:31.803415Z",
     "shell.execute_reply.started": "2023-05-11T13:37:35.493517Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 27 candidates, totalling 135 fits\n",
      "CPU times: user 1min 20s, sys: 3.09 s, total: 1min 24s\n",
      "Wall time: 8min 56s\n",
      "Best parameters: {'max_depth': 30, 'max_features': 'log2', 'min_samples_split': 2, 'n_estimators': 800}\n",
      "Best score: 0.9296704600249261\n"
     ]
    }
   ],
   "source": [
    "# Define the parameter grid for the Random Forest\n",
    "param_grid = {\n",
    "    'n_estimators': [500, 800, 1500],\n",
    "    'max_depth': [10, 20, 30],\n",
    "    'min_samples_split': [2, 5, 10],\n",
    "    'max_features': ['log2']\n",
    "}\n",
    "\n",
    "# Create a Random Forest classifier\n",
    "rf = RandomForestClassifier()\n",
    "\n",
    "# Create a GridSearchCV object\n",
    "grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, verbose = 2, n_jobs = -1)\n",
    "\n",
    "# Fit the GridSearchCV object to the training data\n",
    "%time grid_search.fit(X_train_flat, y_train)\n",
    "\n",
    "# Print the best parameters and the corresponding mean cross-validated score\n",
    "print(\"Best parameters:\", grid_search.best_params_)\n",
    "print(\"Best score:\", grid_search.best_score_) "
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "fcd9fb81-2fe3-4130-bcbc-77f9f583045c",
   "metadata": {},
   "source": [
    "### Predict and evaluate performance on the train set (in-sample)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "e14f1d88-1745-42d6-9613-898f85477b2f",
   "metadata": {},
   "source": [
    "#### Predict on the train set (5-fold cross validation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8afad19e-d358-4557-9ca4-e2cdb7efe5b7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T13:46:31.806250Z",
     "iopub.status.busy": "2023-05-11T13:46:31.806031Z",
     "iopub.status.idle": "2023-05-11T13:57:13.067839Z",
     "shell.execute_reply": "2023-05-11T13:57:13.066746Z",
     "shell.execute_reply.started": "2023-05-11T13:46:31.806229Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Perform 5-fold cross-validation on the rf model to get predictions made on each fold\n",
    "best_model_train = grid_search.best_estimator_\n",
    "y_train_pred = cross_val_predict(best_model_train, X_train_flat, y_train, cv = 5, method='predict')\n",
    "y_train_pred_prob = cross_val_predict(best_model_train, X_train_flat, y_train, cv = 5, method='predict_proba')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "0ecbada8-5de4-4bb9-86be-ca2b8814a903",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T13:57:13.071961Z",
     "iopub.status.busy": "2023-05-11T13:57:13.071512Z",
     "iopub.status.idle": "2023-05-11T14:02:32.670302Z",
     "shell.execute_reply": "2023-05-11T14:02:32.669292Z",
     "shell.execute_reply.started": "2023-05-11T13:57:13.071939Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cross-validated scores: [0.94275583 0.92907801 0.93414387 0.92042575 0.92397364]\n"
     ]
    }
   ],
   "source": [
    "# perform cross validation to check if model is overfitting \n",
    "y_scores = cross_val_score(best_model_train, X_train_flat, y_train, cv = 5)\n",
    "print(f'Cross-validated scores: {y_scores}')"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "f736ff7c-6eba-4146-b24a-6bf2d045f8ee",
   "metadata": {},
   "source": [
    "#### Model evaluation "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "fa1d6d80-b3cf-48ee-9a7e-74a2d947d990",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T14:43:03.062047Z",
     "iopub.status.busy": "2023-05-11T14:43:03.061208Z",
     "iopub.status.idle": "2023-05-11T14:43:03.340907Z",
     "shell.execute_reply": "2023-05-11T14:43:03.340108Z",
     "shell.execute_reply.started": "2023-05-11T14:43:03.061990Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAhgAAAIzCAYAAABC72DlAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAADbCklEQVR4nOzdd3gU5drA4d9sT29AgAChSQmKIBylfBwIHQRFQYoKUhUbKCBH9Ch6VCxHAZWqNAsgIqBwVIoUFUUFBFGKtNATenq2zvv9EbISEyCBDZvy3Ne1F9l3Z2afHZKdZ96qKaUUQgghhBA+ZPB3AEIIIYQofSTBEEIIIYTPSYIhhBBCCJ+TBEMIIYQQPicJhhBCCCF8ThIMIYQQQvicJBhCCCGE8DlJMIQQQgjhc5JgCCGEEMLnJMEQQlxX8+bNQ9M078NkMlGpUiX69u3Lvn37cm3bpk2bXNte/Pjjjz/89AmEEAVh8ncAQoiyae7cudSrVw+73c4PP/zAK6+8wvr169mzZw8RERHe7WrWrMn8+fPz7F+rVq3rGa4QopAkwRBC+MWNN95I06ZNgeyaCo/Hw/jx4/n8888ZNGiQd7uAgACaNWvmrzCFEFdJmkiEEMVCTrJx8uRJP0cihPAFqcEQQhQLCQkJANSpUyfPa263O9dzg8GAwSD3R0IUZ/IXKoTwC4/Hg9vtJj09nVWrVvHyyy/zz3/+kzvuuCPXdjt37sRsNud6DBgwwE9RCyEKSmowhBB+8fd+FfXr1+eLL77AZMr9tVSrVi0++eSTXGVRUVFFHp8Q4tpIgiGE8IsPP/yQ+vXrk5aWxqJFi5g5cyb9+vXj66+/zrWdzWbz9s8QQpQckmAIIfyifv363sQhPj4ej8fDrFmz+Oyzz+jVq5efoxNCXCvpgyGEKBbeeOMNIiIieP7559F13d/hCCGukSQYQohiISIignHjxrF7924WLFjg73CEENdIEgwhRLHx+OOPU61aNf7zn//g8Xj8HY4Q4hpoSinl7yCEEEIIUbpIDYYQQgghfE4SDCGEEEL4nCQYQgghhPA5STCEEEII4XOSYIgyZ968eWia5n2YTCYqVapE37592bdvX777uFwupk+fTvPmzQkLCyMgIID69evz9NNPc/bs2Xz30XWdjz76iPbt21OuXDnMZjMVKlSgW7durFixokBzPTgcDqZMmcL//d//ERERgcViISYmht69e/Ptt99e03kozpxOJ8OHD6dSpUoYjUYaNWpUpO83cODAXL8Tf38UlerVq9OtW7er3v9S8ZYrV+6qjvXCCy9ccbucv59Dhw4VPmBRpshMnqLMmjt3LvXq1cNut/PDDz/wyiuvsH79evbs2UNERIR3u8zMTLp27crGjRt58MEHee655wgICGDTpk28+eabLFiwgDVr1lC3bl3vPna7nR49erB69Wr69u3L9OnTqVixIqdPn2blypXcc889LFq0iDvvvPOS8Z05c4bOnTuzY8cOBg8ezFNPPUVkZCTHjx/niy++oF27dmzdupWbb765SM+TP0yfPp2ZM2fy7rvv0qRJE4KDg4v8PQMCAli3bl2Rv4+v9erVi9GjR+cqM5vNfopGiIsoIcqYuXPnKkBt3rw5V/mLL76oADVnzpxc5Q8++KAC1CeffJLnWH/++acKCwtTDRo0UG6321v+8MMPK0B98MEH+cawd+9e9dtvv102zi5duiiTyaTWrl2b7+u//PKLOnz48GWPUVCZmZk+OY6vDB06VAUEBPj0mJf7jA888IAKCgry6fsVRGxsrLr99tuven9APfrooz6JBVDjx4+/4nY5fz8JCQk+eV9RekkTiRAX5KyLcfLkSW9ZUlISc+bMoVOnTvTp0yfPPnXq1OFf//oXO3fu5PPPP/fuM2vWLDp16nTJZcVvuOEGGjZseMlYtm7dytdff82QIUNo27Ztvtv84x//oFq1agC88MIL+Vbl51ednVMtv3TpUho3bozNZuPFF1+kcePGtGrVKs8xPB4PMTEx3H333d4yp9PJyy+/TL169bBarZQvX55BgwZx+vTpXPuuW7eONm3aEBUVRUBAANWqVaNnz55kZmZe8rNrmsasWbPIysryVvnPmzcPyK4ZGjduHDVq1PA2Fz366KMkJyfnOsalPuO1stvtjB49mkaNGhEWFkZkZCTNmzfniy++yLOtruu8++67NGrUiICAAMLDw2nWrBnLly/Ps+3KlSu55ZZbCAgIoF69esyZM+eaY81x5MgR7r//fipUqIDVaqV+/fq89dZbBWqi++mnn2jZsiU2m43KlSszbtw4XC6Xz2ITpZs0kQhxQUJCApCdNORYv349brebHj16XHK/Hj168Mwzz7BmzRp69uzJ+vXrcblcl93nSlavXu09dlH49ddf2b17N//+97+pUaMGQUFBVK5cmZEjR7Jv3z5uuOGGXLGcOHGCQYMGAdkXzjvvvJPvv/+esWPH0qJFCw4fPsz48eNp06YNW7ZsISAggEOHDnH77bfTqlUr5syZQ3h4OMePH2flypU4nU4CAwPzjW3Tpk289NJLrF+/3ttkUatWLZRS9OjRg7Vr1zJu3DhatWrFjh07GD9+PJs2bWLTpk1YrdbLfsYrcbvdecoMBgMGQ/a9mMPh4Ny5c4wZM4aYmBicTifffPMNd999N3Pnzs2VUA4cOJCPP/6YIUOG8J///AeLxcKvv/6ap+/Cb7/9xujRo3n66aeJjo5m1qxZDBkyhNq1a/PPf/7zijErpfLEbTQa0TSN06dP06JFC5xOJy+99BLVq1fnf//7H2PGjOHAgQNMmzbtksfdtWsX7dq1o3r16sybN4/AwECmTZsm07iLgvN3FYoQ11tOFe9PP/2kXC6XSktLUytXrlQVK1ZU//znP5XL5fJu+9prrylArVy58pLHy8rKUoDq0qVLgfe5kuHDhytA7dmzp0Dbjx8/XuX355xfdXZsbKwyGo3qzz//zLXtmTNnlMViUc8880yu8t69e6vo6GjveVm4cKEC1JIlS3Jtt3nzZgWoadOmKaWU+uyzzxSgtm/fXqDPcLH8mixWrlypAPXGG2/kKl+0aJEC1HvvvXfFz3i59wPyfbRr1+6S+7ndbuVyudSQIUNU48aNveXfffedAtSzzz572feNjY1VNpstV1NXVlaWioyMVA899NAV475UzO+//75SSqmnn35aAernn3/Otd/DDz+sNE3LdX74WxNJnz59VEBAgEpKSsr1eevVqydNJKJApIlElFnNmjXDbDYTEhJC586diYiI4IsvvsBkurqKvaIcbeBrDRs2zFVTAxAVFUX37t354IMPvNXn58+f54svvmDAgAHe8/K///2P8PBwunfvjtvt9j4aNWpExYoV2bBhAwCNGjXCYrHw4IMP8sEHH3Dw4MFrijmnNmPgwIG5yu+55x6CgoJYu3btFT/j5QQEBLB58+Y8j7/f5S9evJiWLVsSHByMyWTCbDYze/Zsdu/e7d3m66+/BuDRRx+94vs2atTI29QFYLPZqFOnDocPHy5Q3L17984Tc07N17p164iLi+PWW2/Ntc/AgQNRSl22U+v69etp164d0dHR3jKj0ZhvU6EQ+ZEEQ5RZH374IZs3b2bdunU89NBD7N69m379+uXaJueLP6f5JD85r1WtWrXA+1yJL45xOZUqVcq3fPDgwRw/fpw1a9YAsHDhQhwOR66L+smTJ0lOTsZisWA2m3M9kpKSOHPmDJDdrPHNN99QoUIFHn30UWrVqkWtWrV4++23ryrms2fPYjKZKF++fK5yTdOoWLFinuHCl/qMl2IwGGjatGmex8VJytKlS+nduzcxMTF8/PHHbNq0ic2bNzN48GDsdrt3u9OnT2M0GqlYseIV3zcqKipPmdVqJSsrq0Bxly9fPk/MOcNUz549m+95qFy5svf1Szl79my+8RfkMwkB0gdDlGH169f3duyMj4/H4/Ewa9YsPvvsM3r16uUtN5lMfP755wwfPjzf4+R07uzQoYN3H7PZfNl9rqRTp04888wzfP7553Tu3PmK29tsNiC7j8DF/RByLvZ/d6nalk6dOlG5cmXmzp1Lp06dmDt3LrfddhtxcXHebcqVK0dUVBQrV67M9xghISHen1u1akWrVq3weDxs2bKFd999lyeeeILo6Gj69u17xc91saioKNxuN6dPn86VZCilSEpK4h//+EeBPuO1+Pjjj6lRowaLFi3KdXyHw5Fru/Lly+PxeEhKSip0ouNLUVFRJCYm5ik/ceIEwGXny4iKiiIpKSlPeX5lQuRHajCEuOCNN94gIiKC559/3ttEULFiRQYPHsyqVatYtGhRnn327t3L66+/ToMGDbzV0hUrVmTo0KGsWrWKDz/8MN/3OnDgADt27LhkLLfccgtdunRh9uzZl6zG3rJlC0eOHAGyR00AeY65YsWKy37mvzMajfTv35/PP/+c77//ni1btjB48OBc23Tr1o2zZ8/i8XjyveO/eD6Qi4972223MXXqVCC7A2ZhtWvXDsi+yF9syZIlZGRkeF8vSpqmYbFYciUXSUlJeUaRdOnSBciez8Of2rVrx65du/Kc7w8//BBN04iPj7/kvvHx8axduzbXqCqPx5Pv34EQ+ZEaDCEuiIiIYNy4cYwdO5YFCxZw//33AzBx4kT+/PNP7r//fr777ju6d++O1Wrlp59+4s033yQkJIQlS5ZgNBq9x5o4cSIHDx5k4MCBrFq1irvuuovo6GjOnDnDmjVrmDt3Lp988sllh6p++OGHdO7cmS5dujB48GC6dOlCREQEiYmJrFixgoULF7J161aqVatG165diYyM9I5YMJlMzJs3j6NHjxb6PAwePJjXX3+de++9l4CAgDxt7n379mX+/Pl07dqVkSNHcuutt2I2mzl27Bjr16/nzjvv5K677mLGjBmsW7eO22+/nWrVqmG3273DL9u3b1/ouDp06ECnTp3417/+RWpqKi1btvSOImncuDH9+/cv9DEvpus6P/30U76vNW7cGKvV6h36+sgjj9CrVy+OHj3KSy+9RKVKlXLNAtuqVSv69+/Pyy+/zMmTJ+nWrRtWq5Vt27YRGBjI448/fk2xFtSTTz7Jhx9+yO23385//vMfYmNj+fLLL5k2bRoPP/zwZfuo/Pvf/2b58uW0bduW559/nsDAQKZOnUpGRsZ1iV2UAv7uZSrE9XapibaUyu7BX61aNXXDDTfkmjjL6XSqqVOnqttuu00FBwcrq9Wq6tatq8aOHavOnDmT7/u43W71wQcfqLZt26rIyEhlMplU+fLlVZcuXdSCBQuUx+O5YqxZWVnqnXfeUc2bN1ehoaHKZDKpypUrq7vvvlt9+eWXubb95ZdfVIsWLVRQUJCKiYlR48ePV7Nmzcp3FMmVJndq0aKFAtR9992X7+sul0u9+eab6uabb1Y2m00FBwerevXqqYceekjt27dPKaXUpk2b1F133aViY2OV1WpVUVFRqnXr1mr58uVX/NyXmvgqKytL/etf/1KxsbHKbDarSpUqqYcfflidP38+13aFncDqcqNIAO9nUip7lFD16tWV1WpV9evXV++//36+o3g8Ho+aNGmSuvHGG5XFYlFhYWGqefPmasWKFVeMs3Xr1qp169ZXjJsCTLR1+PBhde+996qoqChlNptV3bp11X//+988v3/kM9HWDz/8oJo1a6asVquqWLGieuqpp9R7770no0hEgWhKKeWPxEYIIYQQpZf0wRBCCCGEz0mCIYQQQgifkwRDCCGEED4nCYYQQgghfE4SDCGEEEL4nCQYQgghhPA5STCEEEII4XNlbiZPXdc5ceIEISEhJWr1SyGEEMLflFKkpaVRuXJlDIbL11GUuQTjxIkT3lUvhRBCCFF4R48epUqVKpfdpswlGDkrPR49epTQ0FA/RyOEEEKUHKmpqVStWjXXqsmXUuYSjJxmkdDQUEkwhBBCiKtQkC4G0slTCCGEED4nCYYQQgghfE4SDCGEEEL4nCQY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gf4OSQhRyhQ6wahRowbbt2/PU/71118TFxdXqGMtWrSIJ554gmeffZZt27bRqlUrunTpwpEjRy67X0pKCgMGDKBdu3aFer+i5pF5MEQJ0bt3byIiIvjggw/yXQZACCGuVaEHUz711FM8+uij2O12lFL88ssvLFy4kFdffZVZs2YV6lgTJ05kyJAhDB06FMjuKLpq1SqmT5/Oq6++esn9HnroIe69916MRiOff/55YT9CkdE9MkxVlAy33norBw8eJDw83N+hCCFKqULXYAwaNIjx48czduxYMjMzuffee5kxYwZvv/02ffv2LfBxnE4nW7dupWPHjrnKO3bsyI8//njJ/ebOncuBAwcYP358YUMvcrqSJhJRPOm6zsiRI9myZYu3TJILIURRuqr77GHDhjFs2DDOnDmDrutUqFCh0Mc4c+YMHo+H6OjoXOXR0dEkJSXlu8++fft4+umn+f77771TGV+Jw+HA4XB4n6emphY61oK6ePoAaSIRxYXH42Ho0KHMmzePBQsWcODAAUJDQ/0dlhCilLumAe/lypW7quTiYpqW+05fKZWnDLK/JO+9915efPFF6tSpU+Djv/rqq4SFhXkfVatWvaZ4L8ftMVw0TLXI3kaIAvN4PAwZMoR58+ZhNBqZMmWKJBdCiOui0DUYNWrUyDcByHHw4MECHadcuXIYjcY8tRWnTp3KU6sBkJaWxpYtW9i2bZt32Whd11FKYTKZWL16NW3bts2z37hx4xg1apT3eWpqapElGUqGqYpixOPxMGjQID766COMRiMLFy7knnvu8XdYQogyotAJxhNPPJHrucvlYtu2baxcuZKnnnqqwMexWCw0adKENWvWcNddd3nL16xZw5133pln+9DQ0DzzbEybNo1169bx2WefUaNGjXzfx2q1YrVaCxzXtfDkWotEEgzhPx6Ph4EDB/Lxxx9jNBr55JNP6NWrl7/DEkKUIYVOMEaOHJlv+dSpU3N1ICuIUaNG0b9/f5o2bUrz5s157733OHLkCMOHDweyax+OHz/Ohx9+iMFg4MYbb8y1f4UKFbDZbHnK/UWGqYriYtKkSXz88ceYTCY++eQTevbs6e+QhBBljM8WHejSpQtLliwp1D59+vRh8uTJ/Oc//6FRo0Z89913fPXVV8TGxgKQmJh4xTkxipPca5HIVOHCfx599FG6dOnCp59+KsmFEMIvNKWUT66Eb7zxBtOmTePQoUO+OFyRSU1NJSwsjJSUFJ91djuRdIy1cz5k95G7+OqnG9B1D3PnGmnSRCbDENePx+PBYDB4+0hdqsO0EEJcrcJcQwt9BWzcuHGuLy2lFElJSZw+fZpp06YVPtpS5OJhqrIgpbie3G439913H7Vq1eKVV15B07Q8yYXH48HlcvkpQiFESWGxWHyyqnKhE4wePXrkem4wGChfvjxt2rShXr161xxQSSZ9MIQ/uFwu7rvvPhYvXozZbKZ///7Ur1/f+3rOTUBycrL/ghRClBgGg4EaNWpgsViu6TiFSjDcbjfVq1enU6dOVKxY8ZreuDTKnio8u8XJbJaqaVH0XC4X/fr1Y8mSJZjNZpYsWZIruQC8yUWFChUIDAyUZhMhxCXpus6JEydITEykWrVq1/R9UagEw2Qy8fDDD7N79+6rfsPS7OIaDGkiEUXN5XLRt29fli5disViYcmSJXTr1i3XNh6Px5tcREVF+SlSIURJUr58eU6cOIHb7cZsNl/1cQp9GbztttvYtm3bVb9haXbxWiQyD4YoSk6nkz59+niTi2XLluVJLgBvn4vAwMDrHaIQooTKaRrxeDzXdJxC98F45JFHGD16NMeOHaNJkyYEBQXler1hw4bXFFBJ5vFIHwxxfWzYsIFly5ZhtVpZtmwZXbp0uez20iwihCgoX31fFDjBGDx4MJMnT6ZPnz4AjBgxIlcwOUPirjXjKcl0HVmLRFwXHTt2ZNasWcTExNC5c2d/hyOEEHkUuInkgw8+wG63k5CQkOdx8OBB779lmS5ThYsi5HA4OHv2rPf5kCFDJLm4DpKSkujQoQNBQUEFXuL+hRdeoFGjRpfdZuDAgXlG5fnK7Nmz6dixY5Ecu6waM2ZMrhtrcWUFTjBy5uOKjY297KMsk2Gqoqg4HA569epFmzZtOH36tL/DKXIDBw70zuVhMpmoVq0aDz/8MOfPn8+z7Y8//kjXrl2JiIjAZrNx00038dZbb+Vbm7p+/Xq6du1KVFQUgYGBxMXFMXr0aI4fP37JWCZNmkRiYiLbt29n7969Pv2cV/L777/TunVrAgICiImJ4T//+Q9XmhvR4XDw/PPP89xzz+V57dixY1gslnynFDh06BCaprF9+/Y8r/Xo0YOBAwfmKtu/fz+DBg2iSpUqWK1WatSoQb9+/Qq9ZERhLVmyhLi4OKxWK3FxcSxbtuyK+3z66ac0atSIwMBAYmNj+e9//5tnm6lTp1K/fn0CAgKoW7cuH374Ya7Xx44dy9y5c0lISPDZZyntCtXJU9pxLy+7BiP7j18SDOErdrudu+++m//973/s37+fPXv2+Duk66Jz584kJiZy6NAhZs2axYoVK3jkkUdybbNs2TJat25NlSpVWL9+PXv27GHkyJG88sor9O3bN9fFeObMmbRv356KFSuyZMkSdu3axYwZM0hJSeGtt966ZBwHDhygSZMm3HDDDVSoUKHIPu/fpaam0qFDBypXrszmzZt59913efPNN5k4ceJl91uyZAnBwcG0atUqz2vz5s2jd+/eZGZm8sMPP1x1bFu2bKFJkybs3buXmTNnsmvXLpYtW0a9evUYPXr0VR/3SjZt2kSfPn3o378/v/32G/3796d37978/PPPl9zn66+/5r777mP48OH88ccfTJs2jYkTJzJlyhTvNtOnT2fcuHG88MIL7Ny5kxdffJFHH32UFStWeLepUKECHTt2ZMaMGUX2+UodVUCapqnw8HAVERFx2Udxl5KSogCVkpLis2MeTzyqPnzlFTWgR4Jq2NChbrrJrpKTfXZ4UYZlZWWpLl26KEAFBASob775ptD779q1S2VlZRVRhEXjgQceUHfeeWeuslGjRqnIyEjv8/T0dBUVFaXuvvvuPPsvX75cAeqTTz5RSil19OhRZbFY1BNPPJHv+50/fz7f8tjYWEX2XYMC1AMPPKCUUurw4cPqjjvuUEFBQSokJETdc889Kikpybvf+PHj1c033+x97na71ZNPPqnCwsJUZGSkeuqpp9SAAQPyfMaLTZs2TYWFhSm73e4te/XVV1XlypWVruuX3K979+5qzJgxecp1XVc1a9ZUK1euVP/617/UoEGDcr2ekJCgALVt27Y8+955553ez67rumrQoIFq0qSJ8ng8eba91Ln0hd69e6vOnTvnKuvUqZPq27fvJffp16+f6tWrV66ySZMmqSpVqnjPY/PmzfOcs5EjR6qWLVvmKps3b56qWrXqtXyEEuFy3xuFuYYWahTJiy++SFhYmK9znFJBUyrXMFWZB0NcK7vdzl133cXKlSsJCAjgf//7H23btvV3WH5x8OBBVq5cmWtM/urVqzl79ixjxozJs3337t2pU6cOCxcupE+fPixevBin08nYsWPzPf6l+lZs3ryZAQMGEBoayttvv01AQABKKXr06EFQUBDffvstbrebRx55hD59+rBhw4Z8j/PWW28xZ84cZs+eTVxcHG+99RbLli277P/npk2baN26NVar1VvWqVMnxo0bx6FDh6hRo0a++33//ffcd999ecrXr19PZmYm7du3p0qVKtx22228/fbbhISEXDKG/Gzfvp2dO3eyYMGCfKeTvlw/lQkTJjBhwoTLHv/rr7/Ot/YFss/Jk08+mausU6dOTJ48+ZLHczgceYZpBwQEcOzYMQ4fPkz16tVxOBzYbLY82/zyyy+4XC7v792tt97K0aNHOXz4cJnvElAQhUow+vbte12rCEsMpTA43ejaX19+0kQirkVWVhY9evRg9erVBAYG8uWXX9KmTRufHNujKzKcbp8cqzCCLCaMhoI3s/7vf/8jODgYj8eD3W4HyNU8kNMf4u8zl+aoV6+ed5t9+/YRGhpKpUqVChVz+fLlsVqtBAQEeGcvXrNmDTt27CAhIYGqVasC8NFHH9GgQQM2b97MP/7xjzzHmTx5MuPGjfOubDtjxgxWrVp12fdOSkqievXqucqio6O9r+WXYCQnJ5OcnEzlypXzvDZ79mz69u2L0WikQYMG1K5dm0WLFjF06NArn4iL7Nu3D+CqloYYPnw4vXv3vuw2MTExl3wtKSnJew5yREdHk5SUdMl9OnXqxJNPPsnAgQOJj49n//793oQkMTHROzv1rFmz6NGjB7fccgtbt25lzpw5uFwuzpw54/29yYnt0KFDkmAUQIETDOl/cRkOOxrgMfx1Ok2ykKq4BmfPnmXv3r0EBgby1Vdf0bp1a58dO8Pp5peD53x2vIK6tWYkobaCzwoYHx/P9OnTyczMZNasWezdu5fHH388z3bqEp0e1UWrySofriy7e/duqlat6k0uAOLi4ggPD2f37t15EoyUlBQSExNp3ry5t8xkMtG0adMrdtj8e8w521/qs2RlZQHkuRtPTk5m6dKlbNy40Vt2//33M2fOnEInGFeK4XIiIyOJjIws9H4Xy++cXC6WYcOGceDAAbp164bL5SI0NJSRI0fywgsvYLxwJ/jcc8+RlJREs2bNUEoRHR3NwIEDeeONN7zbQHatBkBmZuY1fYayosCXwSv9IZRpF06NNJEIX8nptHjs2DH+7//+z6fHDrKYuLXmtX3JX+37Fmr7oCBq164NwDvvvEN8fDwvvvgiL730EgB16tQBsi/4LVq0yLP/nj17iIuL826bc6EvbC3G313qgubLJAagYsWKee7MT506BZDnLj5HVFQUmqblGW2zYMEC7HY7t912W654dV1n165dxMXFeZu/U1JS8hw3OTnZe8d+8Xm/0lDcv7vWJpJLnZNLnQ/ITkhef/11JkyYQFJSEuXLl2ft2rUA3hqigIAA5syZw8yZMzl58iSVKlXivffeIyQkhHLlynmPde5cdmJevnz5K35WUYhRJLquS/PIFeiyFom4BpmZmbna8KtXr+7z5ALAaNAItZmv+6MwzSP5GT9+PG+++SYnTpwAsicbi4yMzHcEyPLly9m3bx/9+vUDoFevXlgsFt544418j12YlWbj4uI4cuQIR48e9Zbt2rWLlJSUfJtrwsLCqFSpEj/99JO3zO12s3Xr1su+T/Pmzfnuu+9wOp3estWrV1O5cuU8TSc5LBYLcXFx7Nq1K1f57NmzGT16NNu3b/c+fvvtN+Lj45kzZw4AERERlC9fns2bN+faNysri507d1K3bl0AGjVq5O1Hout6nhgudy6HDx+eK4b8Hk2bNr3sOVmzZk2ustWrV+ebYP6d0WgkJiYGi8XCwoULad68eZ5rmtlspkqVKhiNRj755BO6deuWq5/JH3/8gdlspkGDBld8P0HBR5GUFkUyiuTgXrXg+RdUj26nVcOGDnXzzfYr7yTERTIyMlTbtm2VyWRSy5Yt89lxS9MoEqWUatKkiXr00Ue9zxcvXqyMRqMaNmyY+u2331RCQoKaNWuWioiIUL169co12mLq1KlK0zQ1ePBgtWHDBnXo0CG1ceNG9eCDD6pRo0ZdMpaLR1AolT2KonHjxqpVq1Zq69at6ueff1ZNmjRRrVu39m7z91Ekr732moqIiFBLly5Vu3fvVsOGDVMhISGXHUWSnJysoqOjVb9+/dTvv/+uli5dqkJDQ9Wbb7552XM3atQo1bNnT+/zbdu2KUDt3r07z7bvvfeeKl++vHI6nUoppV5//XUVERGhPvzwQ7V//361efNm1atXL1WxYsVc35k///yzCgkJUS1btlRffvmlOnDggPrtt9/Uyy+/rP75z39eNr5r8cMPPyij0ahee+01tXv3bvXaa68pk8mkfvrpJ+827777rmrbtq33+enTp9X06dPV7t271bZt29SIESOUzWZTP//8s3ebP//8U3300Udq79696ueff1Z9+vRRkZGRKiEhIdf7jx8/PtexSytfjSKRBMMHchKM7l3OqIYNHapxY0kwRMGlp6er+Ph4Bajg4GC1ceNGnx27tCUY8+fPVxaLRR05csRb9t1336nOnTursLAwZbFYVFxcnHrzzTeV2+3Os/+aNWtUp06dVEREhLLZbKpevXpqzJgx6sSJE5eM5e8JhlKFH6bqcrnUyJEjVWhoqAoPD1ejRo264jBVpZTasWOHatWqlbJarapixYrqhRdeuOwQVaWU2r17twoICFDJF8bKP/bYYyouLi7fbU+dOqWMRqNasmSJUkopj8ejpk6dqho2bKiCgoJUTEyM6tmzp9q3b1+eff/88081YMAAVblyZWWxWFRsbKzq16+f+vXXXy8b37VavHixqlu3rjKbzapevXre2HOMHz9excbGep+fPn1aNWvWTAUFBanAwEDVrl27XAmJUkrt2rVLNWrUSAUEBKjQ0FB15513qj179uR57zp16qiFCxcWyecqTnyVYGhKla3OFampqYSFhZGSkkJoaKhPjnkiYR/fzlvA/F8e4+iJEMxmxZYt1ivvKMq8jIwMunXrxoYNGwgJCWHlypUFqu4tqJzp/WvUqJGn458ovXr37k3jxo0ZN26cv0MpNb788kueeuopduzYgamU9+K/3PdGYa6h0lPAh/QLq6kajWUqZxNXKT09na5du7JhwwZCQ0ML3JYsxJX897//JTg42N9hlCoZGRnMnTu31CcXviRnyody1iKROTDElWRmZtK1a1e+//57b3JxcQ9/Ia5FbGxsvkN6xdW70vwdIi+pwfAh/cLaSjKCRFyJzWajbt26hIWFsWbNGkkuhBCljlwKfUhqMERBGQwGZs6cyZYtW7j11lv9HY4QQvicJBg+JH0wxOWkpqby4osv4nZnT9NtMBi8E0kJIURpI30wfMijg1JSgyHySk1NpXPnzmzatIkTJ04wc+ZMf4ckhBBFSmowfChnJk/pgyEulpKSQqdOndi0aRMRERE89NBD/g5JCCGKnNRg+FDOrLmSYIgcycnJdOrUiV9++YXIyEi++eYbGjdu7O+whBCiyEmC4UPuC30wTCbpgyGyk4uOHTuyefNmIiMjWbt2baEXhxJCiJJK7rV9KKeTp9RgCKUUPXr0YPPm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9mrI4sokwfARXZpIypxTp07Rrl07/vjjDypWrMj69euv6q7uurMEZ1/s/fG+hRAUFORdq+Wdd94hPj6eF198kZdeegmAOnXqANkX/BYt8n6ePXv2EBcX590250Jf2FqMv7vUBc2XSczVioqKQtO0PKNtFixYgN1uz7Vqr1IKXdfZtWsXcXFxhIVlJ38pKSl5jpucnOy9Y7/4vBd26PWECROYMGHCZbf5+uuv8+2gClCxYsU8tRWnTp3KU6txMU3TeP3115kwYQJJSUmUL1+etWvXAlC9enUAXn31VVq2bMlTTz0FQMOGDQkKCqJVq1a8/PLL3t+Zc+fOAdk1W+LKinE9bslycQ1Gca4dF74zduxY/vjjDypVqsSGDRtKRnIB2c0UtrDr/yhE80h+xo8fz5tvvsmJEycA6NixI5GRkfmOAFm+fDn79u2jX79+APTq1QuLxcIbb7yR77GTk5MLHEdcXBxHjhzh6NGj3rJdu3aRkpKSb+1VWFgYlSpV4qeffvKWud1utm7dWuD3LCiLxUJcXBy7du3KVT579mxGjx7N9u3bvY/ffvuN+Ph45syZA0BERATly5dn8+bNufbNyspi586d1K1bF4BGjRoRFxfHW2+9lW9fo8udy+HDh+eKIb9Hfv1HcjRv3pw1a9bkKlu9enW+CebfGY1GYmJisFgsLFy4kObNm3tHBWVmZuZp1syp3bi4lumPP/7AbDbToEGDK76fQEaR+MLxg3vVe0+95h1FMmxYms+OLYqv8+fPqzvvvFPt2bPH36FcUmkaRaKUUk2aNFGPPvqo9/nixYuV0WhUw4YNU7/99ptKSEhQs2bNUhEREapXr17eUQJKKTV16lSlaZoaPHiw2rBhgzp06JDauHGjevDBB9WoUaMuGcvFIyiUyh5F0bhxY9WqVSu1detW9fPPP6smTZqo1q1be7f5+yiS1157TUVERKilS5eq3bt3q2HDhqmQkJArjgzZuXOn2rZtm+revbtq06aN2rZtW76jPC42atQo1bNnT+/zbdu2KUDt3r07z7bvvfeeKl++vHI6nUoppV5//XUVERGhPvzwQ7V//361efNm1atXL1WxYsVc35k///yzCgkJUS1btlRffvmlOnDggPrtt9/Uyy+/rP75z39eNr5r8cMPPyij0ahee+01tXv3bvXaa68pk8mkfvrpJ+827777rmrb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",
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
    "ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], xlabel=\"False Positive Rate\", ylabel=\"True Positive Rate\", title=\"RF\\nROC Curves for Each Fold\")\n",
    "ax.legend(loc=\"lower right\")\n",
    "\n",
    "# Show the plot\n",
    "plt.show()"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "9fa175d9-54c8-4fc4-9b36-7a232772d4e1",
   "metadata": {},
   "source": [
    "### Predict and evaluate performance on test set (out-of-sample)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "ab6c1345-a982-46c7-81ac-9a160cc94488",
   "metadata": {},
   "source": [
    "#### Predict on test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "09d66901-e75d-459d-b1a6-76977ba1e0b2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T14:02:33.393090Z",
     "iopub.status.busy": "2023-05-11T14:02:33.392886Z",
     "iopub.status.idle": "2023-05-11T14:02:34.736914Z",
     "shell.execute_reply": "2023-05-11T14:02:34.736028Z",
     "shell.execute_reply.started": "2023-05-11T14:02:33.393072Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Use the best model to make predictions on the test set\n",
    "y_test_pred = grid_search.best_estimator_.predict(X_test_flat)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "4a43b619-56b9-4431-8bbc-3834664bbe26",
   "metadata": {},
   "source": [
    "#### Confusion matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "76a8f664-b112-4727-b0cb-b19a6ea51cfb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T14:02:34.738232Z",
     "iopub.status.busy": "2023-05-11T14:02:34.737904Z",
     "iopub.status.idle": "2023-05-11T14:02:34.749306Z",
     "shell.execute_reply": "2023-05-11T14:02:34.748508Z",
     "shell.execute_reply.started": "2023-05-11T14:02:34.738210Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix (Random Forest, Test Data): \n",
      " +----------+-------------+-------------+\n",
      "|          | Predicted 0 | Predicted 1 |\n",
      "+----------+-------------+-------------+\n",
      "| Actual 0 |     1181    |      75     |\n",
      "| Actual 1 |      99     |     1112    |\n",
      "+----------+-------------+-------------+\n"
     ]
    }
   ],
   "source": [
    "# Compute the confusion matrix\n",
    "data = confusion_matrix(y_test, y_test_pred)\n",
    "table = PrettyTable()\n",
    "table.field_names = [\"\", \"Predicted 0\", \"Predicted 1\"]\n",
    "table.add_row([\"Actual 0\", data[0][0], data[0][1]])\n",
    "table.add_row([\"Actual 1\", data[1][0], data[1][1]])\n",
    "print(f\"Confusion Matrix (Random Forest, Test Data): \\n {table}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "830ce3a1-c81f-4334-809c-2d12ca5bf630",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T14:02:34.750581Z",
     "iopub.status.busy": "2023-05-11T14:02:34.750230Z",
     "iopub.status.idle": "2023-05-11T14:02:35.009886Z",
     "shell.execute_reply": "2023-05-11T14:02:35.009161Z",
     "shell.execute_reply.started": "2023-05-11T14:02:34.750559Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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AACK03gAAADFGbUCAIAqJq3KkHgAAADFSTwAAKBKF5FHERIPAACgOI0HAABQnFErAACoYtKqDIkHAABQnMQDAACqeHJ5GRIPAACgOIkHAABUEXiUIfEAAACK03gAAADFGbUCAIAqnlxehsQDAAAoTuIBAABV5B1lSDwAAIDiNB4AAEBxRq0AAKCKJ5eXIfEAAACKk3gAAECVLgKPIiQeAABAceuUeJxzzjnrfMETTjhhvYsBAIBas8ejjHVqPGbNmrVOF6urq9N4AAAAq1mnxmP+/Pml6wAAADqx9d7jsXz58jz88MNZuXJlR9YDAAA1VVdXu1dn1u7G46WXXsrkyZPTu3fvvPvd784TTzyR5NW9Hd/61rc6vEAAAGDj1+7GY+rUqfn973+f3/zmN+nZs2fr8YkTJ+bKK6/s0OIAAODNVldXV7NXZ9bu53hcffXVufLKK7PTTju1+XLe9a535dFHH+3Q4gAAgM6h3YnHc889lyFDhqx2fOnSpZ2+SwMAANZPuxuPHXfcMf/zP//T+vPfm42LL744EyZM6LjKAACgBrrU1e7VmbV71Gr69OnZe++98+CDD2blypU5++yz88ADD+T222/PTTfdVKJGAABgI9fuxGPnnXfOrbfempdeeilbbbVVrr/++jQ0NOT222/P2LFjS9QIAABvGpvLy2h34pEko0ePziWXXNLRtQAAAJ3UejUeq1atylVXXZWHHnoodXV12W677XLAAQekW7f1uhwAAGwwOnfuUDvt7hTuv//+HHDAAWlubs4222yTJHnkkUcyePDgXHPNNRk9enSHFwkAAGzc2r3H46ijjsq73/3uPPXUU7nnnntyzz335Mknn8z222+fz3zmMyVqBAAANnLtTjx+//vf5+67707//v1bj/Xv3z9nnHFGdtxxxw4tDgAA3mxdOvkm71ppd+KxzTbb5Nlnn13t+IIFC/LOd76zQ4oCAAA6l3VKPBYvXtz6v6dNm5YTTjghp59+enbaaackyR133JGvf/3rmTFjRpkqAQDgTSLwKGOdGo9NN920zX2FK5VKDjrooNZjlUolSbLffvtl1apVBcoEAAA2ZuvUeNx4442l6wAAADqxdWo8dtttt9J1AADABqGzP0G8Vtb7iX8vvfRSnnjiiSxfvrzN8e233/4NFwUAAHQu7W48nnvuuXzqU5/KL37xizWet8cD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bGxMr169svvuu+eBBx5o83stLS05/vjjM2jQoPTp0yf7779/nnrqqfWu47VoPAAAYCN211135aKLLsr222/f5viZZ56ZmTNn5rzzzstdd92VoUOHZs8998yLL77YuqapqSlXXXVVrrjiitxyyy1ZsmRJ9t1336xatarD69R4AADARmrJkiU57LDDcvHFF6d///6txyuVSs4666yceuqp+ehHP5pRo0blkksuyUsvvZTLL788SbJo0aL8x3/8R77zne9k4sSJGTNmTC677LLcd999ueGGGzq8Vo0HAABUqaur3au9jjvuuHzkIx/JxIkT2xyfP39+mpubM2nSpNZj9fX12W233XLbbbclSebOnZsVK1a0WdPY2JhRo0a1rulI7moFAAAbiJaWlrS0tLQ5Vl9fn/r6+tXWXnHFFbnnnnty1113rXauubk5SdLQ0NDmeENDQx5//PHWNT169GiTlPx9zd9/vyNJPAAAoEotE4/p06enX79+bV7Tp09frcYnn3wyn//853PZZZelZ8+er/FZ2sYolUpltWP/aF3WrA+NBwAAbCCmTp2aRYsWtXlNnTp1tXVz587NggULMnbs2HTr1i3dunXLTTfdlHPOOSfdunVrTTr+MblYsGBB67mhQ4dm+fLlWbhw4VrXdCSNBwAAbCDq6+uzySabtHmtacxqjz32yH333Zd58+a1vsaNG5fDDjss8+bNy5ZbbpmhQ4dmzpw5rb+zfPny3HTTTdl5552TJGPHjk337t3brHnmmWdy//33t67pSPZ4AABAlbo38DyNN0vfvn0zatSoNsf69OmTgQMHth5vamrKtGnTMnLkyIwcOTLTpk1L7969c+ihhyZJ+vXrl8mTJ+fEE0/MwIEDM2DAgJx00kkZPXr0apvVO4LGAwAAOqGTTz45y5Yty7HHHpuFCxdm/Pjxuf7669O3b9/WNbNmzUq3bt1y0EEHZdmyZdljjz0ye/bsdO3atcPrqatUKpUOv2qNLWnpdB8JeIsbvPc3a10CQIdaduNptS5hrX71x+dr9t57bDuoZu9dmj0eAABAcUatAACgysawx2NjJPEAAACK03gAAADFGbUCAIAqBR7aTSQeAADAm0DiAQAAVWwuL0PiAQAAFKfxAAAAijNqBQAAVbqYtCpC4gEAABQn8QAAgCo2l5ch8QAAAIrTeAAAAMUZtQIAgCqe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kOTV5iFJbrjhhjzzzDP5yU9+kiS5+OKLc+qpp+aMM87IQw89lGnTpuW0007LJZdckiRZunRp9t1332yzzTaZO3duTj/99Jx00knt/k66dOmSc845J/fff38uueSS/PrXv87JJ5/cZs1LL72UM844I5dcckluvfXWLF68OIccckjr+V/+8pf5+Mc/nhNOOCEPPvhgLrzwwsyePbu1uQKAVAA6qSOOOKJywAEHtP78u9/9rjJw4MDKQQcdVKlUKpWvfvWrle7du1cWLFjQuuZXv/pVZZNNNqm8/PLLba611VZbVS688MJKpVKpTJgwoXLMMce0OT9+/PjKe97znjW+9+LFiyv19fWViy++eI11zp8/v5Kkcu+997Y5Pnz48Mrll1/e5tg3vvGNyoQJEyqVSqVy4YUXVgYMGFBZunRp6/kLLrhgjdeqtvnmm1dmzZq11vM/+tGPKgMHDmz9+fvf/34lSeWOO+5oPfbQQw9VklR+97vfVSqVSmXXXXetTJs2rc11Lr300sqwYcNaf05Sueqqq9b6vgB0bvZ4AJ3az3/+87ztbW/LypUrs2LFihxwwAE599xzW89vvvnmGTx4cOvPc+fOzZIlSzJw4MA211m2bFkeffTRJMlDDz2UY445ps35CRMm5MYbb1xjDQ899FBaWlqyxx57rHPdzz33XJ588slMnjw5n/70p1uPr1y5snX/yEMPPZT3vOc96d27d5s62uvGG2/MtGnT8uCDD2bx4sVZuXJlXn755SxdujR9+vRJknTr1i3jxo1r/Z1tt902m266aR566KG8733vy9y5c3PXXXe1SThWrVqVl19+OS+99FKbGgF4a9J4AJ3aBz/4wVxwwQXp3r17GhsbV9s8/ve/WP/dK6+8kmHDhuU3v/nNatda31vK9urVq92/88orryR5ddxq/Pjxbc517do1SVKpVNarnmqPP/54PvzhD+eYY47JN77xjQwYMCC33HJLJk+e3GYkLXn1drj/6O/HXnnllXzta1/LRz/60dXW9OzZ8w3XCcDGT+MBdGp9+vTJO9/5znVev8MOO6S5uTndunXLFltsscY12223Xe6444584hOfaD12xx13rPWaI0eOTK9evfKrX/0qRx111Grne/TokeTVhODvGhoa8va3vz1//vOfc9hhh63xuu9617ty6aWXZtmyZa3NzWvVsSZ33313Vq5cme985zvp0uXVbX8/+tGPVlu3cuXK3H333Xnf+96XJHn44YfzwgsvZNttt03y6vf28MMPt+u7BuCtReMBUGXixImZMGFCDjzwwMyYMSPbbLNNnn766Vx77bU58MADM27cuHz+85/PEUcckXHjxuX9739/fvCDH+SBBx7IlltuucZr9uzZM6ecckpOPvnk9OjRI7vsskuee+65PPDAA5k8eXKGDBmSXr165brrrss73vGO9OzZM/369cvpp5+eE044IZtsskn22WeftLS05O67787ChQszZcqUHHrooTn11FMzefLkfPnLX85jjz2Wf/u3f2vX591qq62ycuXKnHvuudlvv/1y66235rvf/e5q67p3757jjz8+55xzTrp3757Pfe5z2WmnnVobka985SvZd999M3z48PzLv/xLunTpkj/84Q+577778s1vfrP9/yEA6HTc1QqgSl1dXa699tp84AMfyJFHHpmtt946hxxySB577LHWu1AdfPDB+cpXvpJTTjklY8eOzeOPP57Pfvazr3nd0047LSeeeGK+8pWvZLvttsvBBx+cBQsWJHl1/8Q555yTCy+8MI2NjTnggAOSJEcddVS+973vZfbs2Rk9enR22223zJ49u/X2u29729vys5/9LA8++GDGjBmTU089NTNmzGjX533ve9+bmTNnZsaMGRk1alR+8IMfZPr06aut6927d0455ZQceuihmTBhQnr16pUrrrii9fxee+2Vn//855kzZ0523HHH7LTTTpk5c2Y233zzdtUDQOdVV+mIIWEAAIDXIPEAAACK03gAAADFaTwAAIDiNB4AAEBxGg8AAKA4jQcAAFCcxgMAAChO4wEAABSn8QAAAIrTeAAAAMVpPAAAgOI0HgAAQHH/Hwklyi2MXLI3AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x800 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot the confusion matrix using seaborn\n",
    "plt.figure(figsize=(10,8))\n",
    "sns.heatmap(data, annot=True, cmap='Blues', fmt='g')\n",
    "plt.xlabel('Predicted label')\n",
    "plt.ylabel('True label')\n",
    "plt.show()"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "fc91d899-6b0e-4709-a77e-1919fb9db118",
   "metadata": {},
   "source": [
    "#### Performance measures"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "129864d7-863d-426b-81c6-4d1ebe6a3698",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T14:02:35.010979Z",
     "iopub.status.busy": "2023-05-11T14:02:35.010768Z",
     "iopub.status.idle": "2023-05-11T14:02:35.025864Z",
     "shell.execute_reply": "2023-05-11T14:02:35.025123Z",
     "shell.execute_reply.started": "2023-05-11T14:02:35.010960Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Performance Measure Table (Random Forest, Test Data): \n",
      " +---------------------+----------------+-----------------+\n",
      "| Performance Measure | Label 0: fresh | Label 1: rotten |\n",
      "+---------------------+----------------+-----------------+\n",
      "|      Precision      |     0.9227     |      0.9368     |\n",
      "|        Recall       |     0.9403     |      0.9182     |\n",
      "|          F1         |     0.9314     |      0.9274     |\n",
      "|       Support       |      1256      |       1211      |\n",
      "+---------------------+----------------+-----------------+\n",
      "\n",
      "Random Forest Accuracy Score: 0.9295\n"
     ]
    }
   ],
   "source": [
    "# RF performance scores with cross validation\n",
    "rf_scores_test = precision_recall_fscore_support(y_test, y_test_pred)\n",
    "\n",
    "# Create performance measure table\n",
    "tab2 = PrettyTable([\"Performance Measure\", \"Label 0: fresh\", \"Label 1: rotten\"])\n",
    "measure_names = [\"Precision\", \"Recall\", \"F1\", \"Support\"]\n",
    "\n",
    "# Add rows to table\n",
    "tab2.add_row([measure_names[0]] + rf_scores_test[0].round(4).tolist())\n",
    "tab2.add_row([measure_names[1]] + rf_scores_test[1].round(4).tolist())\n",
    "tab2.add_row([measure_names[2]] + rf_scores_test[2].round(4).tolist())\n",
    "tab2.add_row([measure_names[3]] + rf_scores_test[3].round(4).tolist())\n",
    "\n",
    "# Print performance measure table and accuracy\n",
    "print(f\"Performance Measure Table (Random Forest, Test Data): \\n {tab2}\")\n",
    "print(f\"\\nRandom Forest Accuracy Score: {accuracy_score(y_test, y_test_pred).round(4)}\")"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "eb8229bd-07e5-4cdb-8bd7-0ada35b64e68",
   "metadata": {},
   "source": [
    "----"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "51f6e366-ed84-4444-9ae1-8381f7b1c541",
   "metadata": {},
   "source": [
    "### Complexity analysis - Baseline Model Comparison\n",
    "- Baseline = all default RF hyperparameters\n",
    "- Tuned model = optimal hyperparameters found from grid searches"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "fdfadd9c-b0f7-4104-811e-1f9a32a38817",
   "metadata": {},
   "source": [
    "#### Baseline (default RF params)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "5ebc4ddc-7628-4a19-9f95-b959bbaf0b35",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T15:11:08.183880Z",
     "iopub.status.busy": "2023-05-11T15:11:08.182940Z",
     "iopub.status.idle": "2023-05-11T15:13:48.998481Z",
     "shell.execute_reply": "2023-05-11T15:13:48.997285Z",
     "shell.execute_reply.started": "2023-05-11T15:11:08.183800Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 2min 40s, sys: 960 ms, total: 2min 41s\n",
      "Wall time: 2min 40s\n",
      "\n",
      "Performance Measure Table (Baseline Model, Test Data): \n",
      " +---------------------+----------------+-----------------+\n",
      "| Performance Measure | Label 0: fresh | Label 1: rotten |\n",
      "+---------------------+----------------+-----------------+\n",
      "|      Precision      |     0.9235     |      0.9291     |\n",
      "|        Recall       |     0.9323     |      0.9199     |\n",
      "|          F1         |     0.9279     |      0.9245     |\n",
      "|       Support       |      1256      |       1211      |\n",
      "+---------------------+----------------+-----------------+\n",
      "\n",
      "Baseline Model Overall Accuracy Score: 0.9262\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Baseline model\n",
    "baseline_model = RandomForestClassifier()\n",
    "\n",
    "# Fit the baseline model\n",
    "%time baseline_model.fit(X_train_flat, y_train)\n",
    "print()\n",
    "\n",
    "# Evaluate the baseline model\n",
    "y_test_pred_base = baseline_model.predict(X_test_flat)\n",
    "base_scores_test = precision_recall_fscore_support(y_test, y_test_pred_base)\n",
    "\n",
    "# Create performance measure table for the baseline model\n",
    "tab1 = PrettyTable([\"Performance Measure\", \"Label 0: fresh\", \"Label 1: rotten\"])\n",
    "tab1.add_row([measure_names[0]] + base_scores_test[0].round(4).tolist())\n",
    "tab1.add_row([measure_names[1]] + base_scores_test[1].round(4).tolist())\n",
    "tab1.add_row([measure_names[2]] + base_scores_test[2].round(4).tolist())\n",
    "tab1.add_row([measure_names[3]] + base_scores_test[3].round(4).tolist())\n",
    "\n",
    "# Print performance measure table and accuracy for both models\n",
    "print(f\"Performance Measure Table (Baseline Model, Test Data): \\n {tab1}\")\n",
    "print(f\"\\nBaseline Model Overall Accuracy Score: {accuracy_score(y_test, y_test_pred_base).round(4)}\\n\")"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "6d4d0296-c2cf-4268-bb91-db260984c80d",
   "metadata": {},
   "source": [
    "#### Tuned (optimal RF params)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "e6b04899-3927-4d04-8b82-90ea958ff3af",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-11T15:13:49.100523Z",
     "iopub.status.busy": "2023-05-11T15:13:49.100186Z",
     "iopub.status.idle": "2023-05-11T15:15:10.285260Z",
     "shell.execute_reply": "2023-05-11T15:15:10.284284Z",
     "shell.execute_reply.started": "2023-05-11T15:13:49.100505Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 1min 19s, sys: 720 ms, total: 1min 20s\n",
      "Wall time: 1min 19s\n",
      "\n",
      "Performance Measure Table (Random Forest, Test Data): \n",
      " +---------------------+----------------+-----------------+\n",
      "| Performance Measure | Label 0: fresh | Label 1: rotten |\n",
      "+---------------------+----------------+-----------------+\n",
      "|      Precision      |     0.9251     |      0.9409     |\n",
      "|        Recall       |     0.9443     |      0.9207     |\n",
      "|          F1         |     0.9346     |      0.9307     |\n",
      "|       Support       |      1256      |       1211      |\n",
      "+---------------------+----------------+-----------------+\n",
      "\n",
      "Tuned Random Forest Overall Accuracy Score: 0.9327\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Tuned model \n",
    "tuned_model = RandomForestClassifier(max_depth=30, max_features='log2', min_samples_split=2, n_estimators=800)\n",
    "\n",
    "# Fit the complex model\n",
    "%time tuned_model.fit(X_train_flat, y_train)\n",
    "\n",
    "# Evaluate the complex model\n",
    "y_test_pred_tuned = tuned_model.predict(X_test_flat)\n",
    "tuned_scores_test = precision_recall_fscore_support(y_test, y_test_pred_tuned)\n",
    "\n",
    "# Create performance measure table for the tuned model\n",
    "tab2 = PrettyTable([\"Performance Measure\", \"Label 0: fresh\", \"Label 1: rotten\"])\n",
    "tab2.add_row([measure_names[0]] + tuned_scores_test[0].round(4).tolist())\n",
    "tab2.add_row([measure_names[1]] + tuned_scores_test[1].round(4).tolist())\n",
    "tab2.add_row([measure_names[2]] + tuned_scores_test[2].round(4).tolist())\n",
    "tab2.add_row([measure_names[3]] + tuned_scores_test[3].round(4).tolist())\n",
    "\n",
    "print(f\"\\nPerformance Measure Table (Random Forest, Test Data): \\n {tab2}\")\n",
    "print(f\"\\nTuned Random Forest Overall Accuracy Score: {accuracy_score(y_test, y_test_pred_tuned).round(4)}\")"
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
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
