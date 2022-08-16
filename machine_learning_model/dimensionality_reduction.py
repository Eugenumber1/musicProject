import sklearn.decomposition as dc
import sklearn.discriminant_analysis as da

# this method will apply the kernel pca to all features
def kernel_pca(X_train, X_test):
    kpca = dc.KernelPCA(n_components=2, kernel='rbf')
    X_train = kpca.fit_transform(X_train)
    X_test = kpca.transform(X_test)
    return X_train, X_test

# here is the simple pca
def simple_pca(X_train, X_test):
    pca = dc.PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test

# here is the LDA
def lda(X_train, X_test, y_train):
    lda = da.LinearDiscriminantAnalysis(n_components=2)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    return X_train, X_test
