# cython: language_level=3
cimport cython
cimport numpy as np
import numpy as np
from MotionUtils cimport *
from libc.stdio cimport printf

# convert numpy.ndarray with shape (3, 3) to Eigen::Matrix3d
@cython.boundscheck(False)
@cython.wraparound(False)
cdef ndarray_to_eigen_mat3(np.ndarray a, Eigen_Matrix3d & b):
    assert a.size == 9
    cdef int i = 0, j = 0
    cdef np.ndarray[np.float64_t, ndim=2] arr = np.ascontiguousarray(a.reshape((3, 3)))
    for i in range(3):
        for j in range(3):
            b.setValue(i, j, arr[i, j])

# convert Eigen::Matrix3d to numpy.ndarray with shape (3, 3)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=2] eigen_mat3_to_ndarray(const Eigen_Matrix3d & mat):
    cdef np.ndarray[np.float64_t, ndim=2] res = np.zeros((3, 3))
    cdef int i = 0, j = 0
    for i in range(3):
        for j in range(3):
            res[i, j] = mat.getValue(i, j)
    return res


# convert numpy.ndarray with shape (4,) to Eigen::Quaterniond 
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndarray_to_eigen_quat4(np.ndarray a, Eigen_Quaterniond & b):
    assert a.size == 4
    cdef np.ndarray[np.float64_t, ndim=1] arr = np.ascontiguousarray(a.reshape(4))
    b.setValue(arr[0], arr[1], arr[2], arr[3])


# convert Eigen::Quaterniond to numpy.ndarray with shape (4,)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] eigen_quat4_to_ndarray(const Eigen_Quaterniond & a):
    cdef np.ndarray[np.float64_t, ndim=1] res = np.zeros(4)
    res[0] = a.x()
    res[1] = a.y()
    res[2] = a.z()
    res[3] = a.w()
    return res


# convert Eigen::Vector3d to numpy.ndarray with shape (3,)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=2] eigen_vec3_to_ndarray(const Eigen_Vector3d & vec3):
    cdef np.ndarray[np.float64_t, ndim=1] res = np.zeros(3)
    res[0] = vec3.getValue(0)
    res[1] = vec3.getValue(1)
    res[2] = vec3.getValue(2)
    return res


# convert numpy.ndarray with shape (3,) to Eigen::Vector3d
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndarray_to_eigen_vec3(np.ndarray a, Eigen_Vector3d & b):
    assert a.size == 3
    cdef np.ndarray[np.float64_t, ndim=1] res = np.ascontiguousarray( a.reshape(-1))
    b.setValue(0, res[0])
    b.setValue(1, res[1])
    b.setValue(2, res[2])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndarray_to_eigen_MatrixXd(np.ndarray a_, Eigen_MatrixXd & b):
    cdef size_t rows = a_.shape[0]
    cdef size_t cols = a_.shape[1]
    cdef size_t i, j
    cdef np.ndarray[np.float64_t, ndim=2] a = np.ascontiguousarray(a_)
    b.resize(rows,cols)
    for j in range(cols):
        for i in range(rows):
            b.setValue(i,j,a[i, j])

######## std::vector<Eigen::Vector3d> ###############
# convert std::vector<Eigen::Vector3d> to np.ndarray with shape (*, 3)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=2] std_vector_Vector3d_to_ndarray(const std_vector_Vector3d & a):
    cdef np.ndarray[np.float64_t, ndim=2] res = np.zeros((a.size(), 3))
    cdef size_t i = 0
    cdef size_t Size = a.size()
    for i in range(Size):
        res[i, 0] = a.getValue(i, 0)
        res[i, 1] = a.getValue(i, 1)
        res[i, 2] = a.getValue(i, 2)

    return res


# convert std::vector<Eigen::Vector3d> * to np.ndarray with shape (*, 3)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=2] std_vector_Vector3d_ptr_to_ndarray(const std_vector_Vector3d_ptr & a):
    cdef np.ndarray[np.float64_t, ndim=2] res = np.zeros((a.size(), 3))
    cdef size_t i = 0
    cdef size_t Size = a.size()
    for i in range(Size):
        res[i, 0] = a.getValue(i, 0)
        res[i, 1] = a.getValue(i, 1)
        res[i, 2] = a.getValue(i, 2)

    return res


# convert numpy.ndarray with shape (n, 3) to std::vector<Eigen::Vector3d>
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndarray_to_std_vector_Vector3d(np.ndarray a, std_vector_Vector3d & b):
    cdef np.ndarray[np.float64_t, ndim=2] arr = np.ascontiguousarray (a.reshape((-1, 3)))
    cdef size_t size = arr.shape[0]
    cdef size_t i = 0
    b.resize(size)
    for i in range(size):
        b.setValue(i, 0, arr[i, 0])
        b.setValue(i, 1, arr[i, 1])
        b.setValue(i, 2, arr[i, 2])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndarray_to_std_vector_std_vector_Vector3d(np.ndarray a, std_vector[std_vector_Vector3d] & b):
    assert a.ndim == 3 and a.shape[2] == 3
    cdef size_t i, j, size_0 = a.shape[0], size_1 = a.shape[1]
    cdef np.ndarray[np.float64_t, ndim=3] res = np.ascontiguousarray(a)
    b.resize(size_0)
    for i in range(size_0):
        b[i].resize(size_1)
        for j in range(size_1):
            b[i].setValue(j, 0, res[i, j, 0])
            b[i].setValue(j, 1, res[i, j, 1])
            b[i].setValue(j, 2, res[i, j, 2])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=3] std_vector_std_vector_Vector3d_to_ndarray(std_vector[std_vector_Vector3d] & b):
    cdef size_t size_0 = b.size(), size_1 = b[0].size(), i, j
    cdef np.ndarray[np.float64_t, ndim=3] res = np.zeros((size_0, size_1, 3))
    for i in range(size_0):
        for j in range(size_1):
            res[i, j, 0] = b[i].getValue(j, 0)
            res[i, j, 1] = b[i].getValue(j, 1)
            res[i, j, 2] = b[i].getValue(j, 2)
    return res

###### end std::vector<Eigen::Vector3d> ##########

###### std::vector<Eigen::Quaterniond> ##########
# convert std::vector<Eigen::Quaterniond> to numpy.ndarray with shape (n, 4)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=2] std_vector_Quaterniond_to_ndarray(const std_vector_Quaterniond & a):
    cdef np.ndarray[np.float64_t, ndim=2] res = np.zeros((a.size(), 4))
    cdef size_t i = 0, Size = a.size()
    for i in range(Size):
        res[i, 0] = a.getX(i)
        res[i, 1] = a.getY(i)
        res[i, 2] = a.getZ(i)
        res[i, 3] = a.getW(i)

    return res

# convert std::vector<Eigen::Quaterniond> * to numpy.ndarray with shape (n, 4)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=2] std_vector_Quaterniond_ptr_to_ndarray(const std_vector_Quaterniond_ptr & a):
    cdef np.ndarray[np.float64_t, ndim=2] res = np.zeros((a.size(), 4))
    cdef size_t i = 0, Size = a.size()
    for i in range(Size):
        res[i, 0] = a.getX(i)
        res[i, 1] = a.getY(i)
        res[i, 2] = a.getZ(i)
        res[i, 3] = a.getW(i)

    return res


# convert numpy.ndarray with shape (n,4) to std::vector<Eigen::Quaterniond>
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndarray_to_std_vector_Quaterniond(np.ndarray a, std_vector_Quaterniond & b):
    cdef np.ndarray[np.float64_t, ndim=2] res = np.ascontiguousarray (a.reshape((-1, 4)))
    cdef size_t i = 0, Size = res.shape[0]
    b.resize(Size)
    for i in range(Size):
        b.setValue(i, res[i, 0], res[i, 1], res[i, 2], res[i, 3])

##### end std::vector<Eigen::Quaterniond> ########

# convert numpy.ndarray with shape (batch, n, 4) to std::vector<std::vector<Eigen::Quaterniond>>>
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndarray_to_std_vector_std_vector_Quaterniond(np.ndarray a, std_vector_std_vector_Quaterniond & b):
    assert a.ndim == 3 and a.shape[2] == 4
    cdef np.ndarray[np.float64_t, ndim=3] res = np.ascontiguousarray(a)
    cdef size_t i = 0, j = 0, size_0 = res.shape[0], size_1 = res.shape[1]
    b.resize(size_0, size_1)
    for i in range(size_0):
        for j in range(size_1):
            b.setValue(i, j, res[i, j, 0], res[i, j, 1], res[i, j, 2], res[i, j, 3])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=3] std_vector_std_vector_Quaterniond_to_ndarray(const std_vector_std_vector_Quaterniond & a):
    cdef size_t size_0 = a.size_0()
    cdef size_t size_1 = a.size_1()
    cdef size_t i, j
    cdef np.ndarray[np.float64_t, ndim=3] res = np.zeros((size_0, size_1, 4))
    for i in range(size_0):
        for j in range(size_1):
            res[i, j, 0] = a.getX(i, j)
            res[i, j, 1] = a.getY(i, j)
            res[i, j, 2] = a.getZ(i, j)
            res[i, j, 3] = a.getW(i, j)
    return res

##### std::vector<Eigen::Matrix3d> ###
# convert std::vector<Eigen::Matrix3d> to numpy.ndarray with shape (n, 3, 3)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=3] std_vector_Matrix3d_to_ndarray(const std_vector_Matrix3d & a):
    cdef np.ndarray[np.float64_t, ndim=3] res = np.zeros((a.size(), 3, 3))
    cdef size_t i = 0, Size = a.size(), j, k
    for i in range(Size):
        for j in range(3):
            for k in range(3):
                res[i, j, k] = a.getValue(i, j, k)

    return res

# convert std::vector<Eigen::Matrix3d> to numpy.ndarray with shape (n, 3, 3)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=3] std_vector_Matrix3d_ptr_to_ndarray(const std_vector_Matrix3d_ptr & a):
    cdef np.ndarray[np.float64_t, ndim=3] res = np.zeros((a.size(), 3, 3))
    cdef size_t i = 0, Size = res.shape[0], j, k
    for i in range(Size):
        for j in range(3):
            for k in range(3):
                res[i, j, k] = a.getValue(i, j, k)
    return res

# convert numpy.ndarray with shape (n, 3, 3) to std::vector<Eigen::Matrix3d>
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndarray_to_std_vector_Matrix3d(np.ndarray a, std_vector_Matrix3d & b):
    cdef np.ndarray[np.float64_t, ndim=3] arr = np.ascontiguousarray (a.reshape(-1, 3, 3))
    cdef size_t i = 0, Size = arr.shape[0]
    b.resize(Size)
    for i in range(Size):
        b.setValue(i, arr[i, 0, 0], arr[i, 0, 1], arr[i, 0, 2],
                      arr[i, 1, 0], arr[i, 1, 1], arr[i, 1, 2],
                      arr[i, 2, 0], arr[i, 2, 1], arr[i, 2, 2])

##### end std::vector<Eigen::Matrix3d> ###


###### Eigen::MatrixXd
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=2] EigenMatrixXdToNumpy(Eigen_MatrixXd & data):
    cdef size_t i = 0, j = 0, shape0 = data.rows(), shape1 = data.cols()
    cdef np.ndarray[np.float64_t, ndim=2] res = np.zeros((shape0, shape1))
    for i in range(shape0):
        for j in range(shape1):
            res[i, j] = data.getValue(i, j)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void EigenMatrixXdFromNumpy(np.ndarray a, Eigen_MatrixXd & data):
    assert a.ndim == 2
    cdef np.ndarray[np.float64_t, ndim=2] arr = np.ascontiguousarray (a)
    cdef size_t i = 0, j = 0
    cdef size_t shape0 = arr.shape[0], shape1 = arr.shape[1]
    data.resize(shape0, shape1)
    for i in range(shape0):
        for j in range(shape1):
            data.setValue(i, j, arr[i, j])

###### end Eigen::MatrixXd

###### Eigen::VectorXd
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] EigenVectorXdToNumpy(Eigen_VectorXd & data):
    cdef size_t i = 0, shape0 = data.size()
    cdef np.ndarray[np.float64_t, ndim=1] res = np.zeros(shape0)
    for i in range(shape0):
        res[i] = data.getValue(i)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void EigenVectorXdFromNumpy(np.ndarray a, Eigen_VectorXd & data):
    assert a.ndim == 1
    cdef size_t i = 0, shape0 = a.shape[0]
    data.resize(shape0)
    cdef np.ndarray[np.float64_t, ndim=1] res = np.ascontiguousarray (a)
    for i in range(shape0):
        data.setValue(i, res[i])

###### end Eigen::VectorXd

# simple wrapper of Eigen::MatrixXd
cdef class PyEigenMatrixXd:
    cdef Eigen_MatrixXd data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, np.ndarray a):
        EigenMatrixXdFromNumpy(a, (self.data))

    def __init__(self, np.ndarray a):
        pass

    def rows(self) -> int:
        return self.data.rows()

    def cols(self) -> int:
        return self.data.cols()

    def ToNumpy(self) -> np.ndarray:
        return EigenMatrixXdToNumpy((self.data))

    # cdef size_t data_handle(self):
    #    return <size_t> (&(self.data))

    def test1_wrapper(self):  # For test
        """
        void MatrixXd_Test1(MatrixXd& a);
        void MatrixXd_Test1_Wrapper(Eigen_MatrixXd * ptr)
        {
            MatrixXd_Test1(ptr->data);
        }
        """
        MatrixXd_Test1_Wrapper(&(self.data))

# print std::vector<std::string>
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void print_std_vector_std_string(const std_vector[std_string] & res):
    cdef size_t i = 0
    for i in range(res.size()):
        printf("%s\n", res[i].c_str())


# print std::vector<double>
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void print_std_vector_double(const std_vector[double] & res):
    cdef size_t i = 0
    for i in range(res.size()):
        printf("%lf ", res[i])
    printf("\n")


cdef void py_list_int_to_std_vector_int(list a, std_vector[int] & res):
    cdef size_t i = 0, cnt = len(a)
    res.resize(cnt)
    for i in range(cnt):
        res[i] = a[i]

# convert python List[str] to std::vector<std::string>
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void py_list_str_to_std_vector_str(list a, std_vector[std_string] & res):
    cdef size_t i = 0, cnt = len(a), j = 0
    cdef bytes b
    cdef str s
    cdef std_string std_str
    res.resize(cnt)
    for i in range(cnt):
        s = <str?>a[i]
        b = s.encode('ascii')
        std_str = std_string(b)
        res[i] = std_str

def test_py_list_str_to_std_vector_str(list a):
    # a: List[str]
    cdef std_vector[std_string] res
    py_list_str_to_std_vector_str(a, res)
    print_std_vector_std_string(res)

# convert np.ndarray to std::vector<double>
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndarray_to_std_vector_double(np.ndarray a, std_vector[double] & b):
    cdef np.ndarray[np.float64_t, ndim=1] arr = np.ascontiguousarray(a.reshape(-1))
    cdef size_t i = 0, cnt = a.size
    b.resize(cnt)
    for i in range(cnt):
        b[i] = a[i]

# convert np.ndarray to std::vector<int>
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndarray_to_std_vector_int(np.ndarray a, std_vector[int] & b):
    cdef np.ndarray[np.int32_t, ndim=1] arr = np.ascontiguousarray(a.reshape(-1), np.int32)
    cdef size_t i = 0, cnt = a.size
    b.resize(cnt)
    for i in range(cnt):
        b[i] = a[i]


# convert std::vector<int> to np.ndarray
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int32_t, ndim=1] std_vector_int_to_ndarray(const std_vector[int] & a):
    cdef np.ndarray[np.int32_t, ndim=1] res = np.empty(a.size(), np.int32)
    cdef size_t i = 0, cnt = a.size()
    for i in range(cnt):
        res[i] = a[i]
    return res

# convert std::vector<double> to np.ndarray
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] std_vector_double_to_ndarray(const std_vector[double] & a):
    cdef np.ndarray[np.float64_t, ndim=1] res = np.empty(a.size())
    cdef size_t i = 0, cnt = a.size()
    for i in range(cnt):
        res[i] = a[i]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=2] std_vector_std_vector_double_to_ndarray(const std_vector[std_vector[double]] & a):
    assert a.size() > 0
    cdef size_t size_0 = a.size(), size_1 = a[0].size(), i, j
    cdef np.ndarray[np.float64_t, ndim=2] res = np.empty((size_0, size_1))
    for i in range(size_0):
        for j in range(size_1):
            res[i, j] = a[i][j]
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndarray_to_std_vector_std_vector_double(np.ndarray a, std_vector[std_vector[double]] & b):
    assert a.ndim == 2
    cdef np.ndarray[np.float64_t, ndim=2] res = np.ascontiguousarray(a)
    cdef size_t i = 0, j = 0, size_0 = a.shape[0], size_1 = a.shape[1]
    b.resize(size_0)
    for i in range(size_0):
        b[i].resize(size_1)
        for j in range(size_1):
            b[i][j] = res[i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] Eigen_MatrixXd_to_ndarray(const Eigen_MatrixXd & a):
    cdef size_t rows = a.rows()
    cdef size_t cols = a.cols()
    cdef np.ndarray[np.float64_t, ndim=2] res = np.zeros([rows,cols])
    cdef size_t i, j
    for i in range(rows):
        for j in range(cols):
            res[i][j] = a.getValue(i,j)
    return res

# Wrapper of Eigen::VectorXd
cdef class PyEigenVectorXd:
    cdef Eigen_VectorXd data

    def __cinit__(self, np.ndarray a):
        EigenVectorXdFromNumpy(a, self.data)

    def __init__(self, np.ndarray a):
        # self.__cinit__ is called automatically
        pass

    def size(self) -> int:
        return self.data.size()

    def ToNumpy(self) -> np.ndarray:
        return EigenVectorXdToNumpy(self.data)

    cdef Eigen_VectorXd * data_ptr(self):
        return &(self.data)


# Wrapper of Eigen::ArrayXd
cdef class PyEigenArrayXd:
    cdef Eigen_ArrayXd data

    def __cinit__(self, np.ndarray a):
        pass

    def __init__(self, np.ndarray a):
        raise NotImplementedError

    def size(self) -> int:
        pass

    def ToNumpy(self) -> np.ndarray:
        pass

# Wrapper of Eigen::ArrayXXd
cdef class PyEigenArrayXXd:
    cdef Eigen_ArrayXXd data

    def __cinit__(self, np.ndarray a):
        pass

    def __init__(self, np.ndarray a):
        raise NotImplementedError

    def size(self) -> int:
        pass

    def ToNumpy(self) -> np.ndarray:
        pass

cdef void assert_ndarray_mat33(np.ndarray a, int shape0):
    assert a.dtype == np.float64 and a.ndim == 3 and a.shape[0] == shape0 and a.shape[1] == 3 and a.shape[2] == 3

cdef void assert_ndarray_vec3(np.ndarray a, int shape0):
    assert a.dtype == np.float64 and a.ndim == 2 and a.shape[0] == shape0 and a.shape[1] == 3

cdef void assert_ndarray_int32_ndim1(np.ndarray a, int shape0):
    assert a.dtype == np.int32 and a.ndim == 1 and a.size == shape0

cdef class InvDynForceBatchRes:
    cdef np.ndarray _qs
    cdef np.ndarray _dqs
    cdef np.ndarray _ddqs

    cdef np.ndarray _joint_rots # in shape (batch, num joint, 4)
    cdef np.ndarray _linvel # in shape (batch, num joint, 3)
    cdef np.ndarray _angvel # in shape (batch, num joint, 3)
    cdef np.ndarray _linacc # in shape (batch, num joint, 3)
    cdef np.ndarray _angacc # in shape (batch, num joint, 3)
    cdef np.ndarray _f_local # in shape (batch, num joint, 3)
    cdef np.ndarray _t_local # in shape (batch, num joint, 3)

    cdef np.ndarray _com_mass # in shape (batch, )
    cdef np.ndarray _com_pos # in shape (batch, 3)
    cdef np.ndarray _com_linvel # in shape (batch, 3)
    cdef np.ndarray _com_ang_momentums # in shape (batch, 3)
    cdef np.ndarray _com_inertia # in shape (batch, 3, 3)

    def __cinit__(self):
        pass

    def __init__(self):
        pass

    def __len__(self):
        return self._qs.shape[0]

    cpdef sub_seq(self, start_: int=None, end_: int=None, is_copy: bool=False):
        """
        param: 
        start_: Optional[int] = None
        end_: Optional[int] = None
        is_copy: bool = False
        """
        cdef InvDynForceBatchRes res = InvDynForceBatchRes()
        cdef size_t start = start_ if start_ is not None else 0
        cdef size_t end = end_ if end_ is not None else self._qs.shape[0]
        if self._qs is not None:
            res._qs = self._qs[start:end].copy() if is_copy else self._qs[start:end]
        if self._dqs is not None:
            res._dqs = self._dqs[start:end].copy() if is_copy else self._dqs[start:end]
        if self._ddqs is not None:
            res._ddqs = self._ddqs[start:end].copy() if is_copy else self._ddqs[start:end]
        # TODO: _qs_unslice, rots_unslice
        if self._joint_rots is not None:
            res._joint_rots = self._joint_rots[start:end].copy() if is_copy else self._joint_rots[start:end]
        if self._linvel is not None:
            res._linvel = self._linvel[start:end].copy() if is_copy else self._linvel[start:end]
        if self._angvel is not None:
            res._angvel = self._angvel[start:end].copy() if is_copy else self._angvel[start:end]
        if self._angacc is not None:
            res._angacc = self._angacc[start:end].copy() if is_copy else self._angacc[start:end]
        if self._f_local is not None:
            res._f_local = self._f_local[start:end].copy() if is_copy else self._f_local[start:end]
        if self._t_local is not None:
            res._t_local = self._t_local[start:end].copy() if is_copy else self._t_local[start:end]
        # TODO: _f_local_unslice, _t_local_unslice, _com_pos_unslice
        if self._com_mass is not None:
            res._com_mass = self._com_mass[start:end].copy() if is_copy else self._com_mass[start:end]
        if self._com_pos is not None:
            res._com_pos = self._com_pos[start:end].copy() if is_copy else self._com_pos[start:end]
        if self._com_linvel is not None:
            res._com_linvel = self._com_linvel[start:end].copy() if is_copy else self._com_linvel[start:end]
        if self._com_ang_momentums is not None:
            res._com_ang_momentums = self._com_ang_momentums[start:end].copy() if is_copy else self._com_ang_momentums[start:end]
        if self._com_inertia is not None:
            res._com_inertia = self._com_inertia[start:end].copy() if is_copy else self._com_inertia[start:end]

        return res

    @property
    def qs(self) -> np.ndarray:
        return self._qs

    @property
    def dqs(self) -> np.ndarray:
        return self._dqs

    @property
    def ddqs(self) -> np.ndarray:
        return self._ddqs

    @property
    def joint_rots(self) -> np.ndarray:
        return self._joint_rots

    @property
    def linvel(self) -> np.ndarray:
        return self._linvel

    @property
    def angvel(self) -> np.ndarray:
        return self._angvel

    @property
    def linacc(self) -> np.ndarray:
        return self._linacc

    @property
    def angacc(self) -> np.ndarray:
        return self._angacc

    @property
    def f_local(self) -> np.ndarray:
        return self._f_local

    @property
    def t_local(self) -> np.ndarray:
        return self._t_local

    @property
    def com_mass(self) -> np.ndarray:
        return self._com_mass

    @property
    def com_pos(self) -> np.ndarray:
        return self._com_pos

    @property
    def com_linvel(self) -> np.ndarray:
        return self._com_linvel

    @property
    def com_ang_momentums(self) -> np.ndarray:
        return self._com_ang_momentums

    @property
    def com_inertia(self) -> np.ndarray:
        return self._com_inertia


cdef class DInverseDynamics:
    """
    Params:
    np.ndarray body_mass, shape=(body_count,)
    np.ndarray body_inertia, shape=(body_count, 3, 3)
    np.ndarray body_position, shape=(body_count, 3)
    np.ndarray body_rotation, shape=(body_count, 3, 3)
    np.ndarray parent_joint_dof, np.int32, shape=(body_count,)
    np.ndarray parent_joint_pos, shape=(body_count, 3)
    list parent_joint_euler_order, list[str], len == body_count
    np.ndarray parent_joint_euler_axis, shape=(body_count)
    np.ndarray parent_body_index, np.int32
    """
    cdef CInverseDynamicsPtr pointer

    def __cinit__(self,
                 np.ndarray[np.float64_t, ndim=1] body_mass,
                 np.ndarray[np.float64_t, ndim=3] body_inertia,
                 np.ndarray[np.float64_t, ndim=2] body_position,
                 np.ndarray[np.float64_t, ndim=3] body_rotation,
                 np.ndarray[np.int32_t, ndim=1] parent_joint_dof,
                 np.ndarray[np.float64_t, ndim=2] parent_joint_pos,
                 list parent_joint_euler_order,
                 np.ndarray[np.float64_t, ndim=3] parent_joint_euler_axis,
                 np.ndarray[np.int32_t, ndim=1] parent_body_index):
        assert body_mass.ndim == 1
        cdef int body_cnt = body_mass.size

        assert_ndarray_mat33(body_inertia, body_cnt)
        assert_ndarray_vec3(body_position, body_cnt)
        assert_ndarray_mat33(body_rotation, body_cnt)
        assert_ndarray_int32_ndim1(parent_joint_dof, body_cnt)
        assert_ndarray_vec3(parent_joint_pos, body_cnt)
        assert len(parent_joint_euler_order) == body_cnt
        assert_ndarray_mat33(parent_joint_euler_axis, body_cnt)
        assert_ndarray_int32_ndim1(parent_body_index, body_cnt)

        # convert from numpy.ndarray to C++ eigen format
        cdef std_vector[double] body_mass_
        ndarray_to_std_vector_double(body_mass, body_mass_)

        cdef std_vector_Matrix3d body_inertia_
        ndarray_to_std_vector_Matrix3d(body_inertia, body_inertia_)

        cdef std_vector_Vector3d body_position_
        ndarray_to_std_vector_Vector3d(body_position, body_position_)

        cdef std_vector_Matrix3d body_rotation_
        ndarray_to_std_vector_Matrix3d(body_rotation, body_rotation_)

        cdef std_vector[int] parent_joint_dof_
        ndarray_to_std_vector_int(parent_joint_dof, parent_joint_dof_)

        cdef std_vector_Vector3d parent_joint_pos_
        ndarray_to_std_vector_Vector3d(parent_joint_pos, parent_joint_pos_)

        cdef std_vector[std_string] parent_joint_euler_order_
        py_list_str_to_std_vector_str(parent_joint_euler_order, parent_joint_euler_order_)

        cdef std_vector_Matrix3d parent_joint_euler_axis_
        ndarray_to_std_vector_Matrix3d(parent_joint_euler_axis, parent_joint_euler_axis_)

        cdef std_vector[int] parent_body_index_
        ndarray_to_std_vector_int(parent_body_index, parent_body_index_)

        self.pointer = InvDynCreate(&body_mass_,
                                    &body_inertia_,
                                    &body_position_,
                                    &body_rotation_,
                                    &parent_joint_dof_,
                                    &parent_joint_pos_,
                                    &parent_joint_euler_order_,
                                    &parent_joint_euler_axis_,
                                    &parent_body_index_)

    def __init__(self, np.ndarray body_mass, np.ndarray body_inertia, np.ndarray body_position,
                 np.ndarray body_rotation, np.ndarray parent_joint_dof, np.ndarray parent_joint_pos,
                 list parent_joint_euler_order, np.ndarray parent_joint_euler_axis, np.ndarray parent_body_index):
        pass

    def __dealloc__(self):
        self.destroy_immediate()

    def __len__(self):
        if self.pointer != NULL:
            return self.pointer.size()
        else:
            raise ValueError("self.pointer has been deconstructed.")

    def __eq__(self, DInverseDynamics other):
        return self.pointer == other.pointer

    cpdef size_t pointer_handle(self):
        return <size_t> self.pointer

    def GeneralizedCoordinatesDimension(self) -> int:
        """
        return: int
        """
        return InvDynGeneralizedCoordinatesDimension(self.pointer)

    def ConvertToGeneralizeCoordinates(self, np.ndarray[np.float64_t, ndim=1] root_pos, np.ndarray[np.float64_t, ndim=2] v_joint_rots) -> np.ndarray:
        """
        param:
        root_pos: np.ndarray in shape (3,)
        vJointRots: np.ndarray in shape (n, 4)
        return:
        """
        cdef Eigen_Vector3d root_pos_
        ndarray_to_eigen_vec3(root_pos, root_pos_)
        cdef std_vector_Quaterniond v_joint_rots_
        ndarray_to_std_vector_Quaterniond(v_joint_rots, v_joint_rots_)
        cdef std_vector[double] q
        InvDynConvertToGeneralizeCoordinates(self.pointer, &root_pos_, &v_joint_rots_, &q)
        return std_vector_double_to_ndarray(q)

    def ConvertToGeneralizeCoordinatesBatch(
        self, np.ndarray[np.float64_t, ndim=2] root_pos,
        np.ndarray[np.float64_t, ndim=3] v_joint_rots
    ) -> np.ndarray:
        """
        root_pos: np.ndarray in shape (batch, 3,)
        vJointRots: np.ndarray in shape (batch, n, 4), joint order same as character in local coordinate. Root is also included.
        return: q in shape(batch, total dof)
        """
        assert root_pos.shape[0] == v_joint_rots.shape[0]
        cdef std_vector_std_vector_Quaterniond r
        ndarray_to_std_vector_std_vector_Quaterniond(v_joint_rots, r)
        cdef std_vector_Vector3d p
        ndarray_to_std_vector_Vector3d(root_pos, p)
        cdef std_vector[std_vector[double]] q
        InvDynConvertToGeneralizeCoordinatesBatch(self.pointer, &p, &r, &q)
        return std_vector_std_vector_double_to_ndarray(q)

    def ConvertToJointRotations(self, np.ndarray[np.float64_t, ndim=1] q):
        """
        Param: q
        return:
        root_pos: np.ndarray with shape (3,)
        vJointRots: np.ndarray with shape (*, 4)
        """
        cdef std_vector[double] q_
        ndarray_to_std_vector_double(q, q_)
        cdef std_vector_Quaterniond v_joint_rots_
        cdef Eigen_Vector3d root_pos_
        InvDynConvertToJointRotations(self.pointer, &q_, &root_pos_, &v_joint_rots_)
        return eigen_vec3_to_ndarray(root_pos_), std_vector_Quaterniond_to_ndarray(v_joint_rots_)

    def InitializeInverseDynamics(self, np.ndarray[np.float64_t, ndim=1] q, np.ndarray[np.float64_t, ndim=1] qdot, np.ndarray[np.float64_t, ndim=1] qdotdot):
        """
        Param:
        q: np.ndarray
        qdot: np.ndarray
        qdotdot: np.ndarray
        """
        cdef std_vector[double] q_, qdot_, qdotdot_
        ndarray_to_std_vector_double(q, q_)
        ndarray_to_std_vector_double(qdot, qdot_)
        ndarray_to_std_vector_double(qdotdot, qdotdot_)
        InvDynInitializeInverseDynamics(self.pointer, &q_, &qdot_, &qdotdot_)

    def ComputeVelocityAcceleration(self, double gravity_x, double gravity_y, double gravity_z):
        """
        Param: double gravity_x, double gravity_y, double gravity_z
        return: None
        """
        InvDynComputeVelocityAcceleration(self.pointer, gravity_x, gravity_y, gravity_z)

    def ComputeForceTorque(self, np.ndarray[np.float64_t] force, np.ndarray[np.float64_t] torque):
        """
        Param:
        force: Optional, np.ndarray in shape (n, 3)
        torque: Optional, np.ndarray in shape (n, 3)

        return: None
        """
        cdef std_vector_Vector3d force_, torque_
        if force is not None and torque is not None:
            assert force.shape == torque.shape
            assert force.shape[0] == self.pointer.size()

            ndarray_to_std_vector_Vector3d(force, force_)
            ndarray_to_std_vector_Vector3d(torque, torque_)
        InvDynComputeForceTorque(self.pointer, &force_, &torque_)

    def ComputeForceTorqueMomentumsBatch(self,
                                         np.ndarray[np.float64_t, ndim=2] q,
                                         np.ndarray[np.float64_t, ndim=2] dq,
                                         np.ndarray[np.float64_t, ndim=2] ddq,
                                         np.ndarray[np.float64_t, ndim=3] rots,
                                         np.ndarray[np.float64_t, ndim=1] gravity,
                                         np.ndarray[np.float64_t, ndim=2] force,
                                         np.ndarray[np.float64_t, ndim=2] torque
                                         ) -> InvDynForceBatchRes:
        """
        Param:
        q: np.ndarray in shape (batch, dof)
        dq: np.ndarray in shape (batch, dof)
        ddq: np.ndarray in shape (batch, dof)
        rots: np.ndarray in shape (batch, num body, 4)
        force: Optional, np.ndarray in shape (num body, 3)
        torque: Optional, np.ndarray in shape (num body, 3)

        return: InvDynForceBatchRes
        """
        assert q.shape[0] == rots.shape[0]

        cdef std_vector[std_vector[double]] q_, dq_, ddq_
        ndarray_to_std_vector_std_vector_double(q, q_)
        ndarray_to_std_vector_std_vector_double(dq, dq_)
        ndarray_to_std_vector_std_vector_double(ddq, ddq_)
        cdef Eigen_Vector3d gravity_
        ndarray_to_eigen_vec3(gravity, gravity_)
        cdef std_vector[std_vector_Vector3d] linvels, angvels, linaccs, angaccs, fs_in, ts_in, f_local, t_local
        if force is not None:
            ndarray_to_std_vector_std_vector_Vector3d(force, fs_in)
        else:
            fs_in.resize(q_.size())
        if torque is not None:
            ndarray_to_std_vector_std_vector_Vector3d(torque, ts_in)
        else:
            ts_in.resize(q_.size())

        cdef std_vector[double] com_mass
        cdef std_vector_Vector3d com_pos, com_linvel, com_ang_momentums
        cdef std_vector_Matrix3d com_inertia

        InvDynComputeForceTorqueBatch(
            self.pointer, &q_, &dq_, &ddq_, &gravity_, &linvels, &angvels, &linaccs, &angaccs, &fs_in, &ts_in,
            &f_local, &t_local, &com_mass, &com_pos, &com_linvel, &com_ang_momentums, &com_inertia)

        cdef InvDynForceBatchRes result = InvDynForceBatchRes()
        result._qs = q
        result._dqs = dq
        result._ddqs = ddq

        result._joint_rots = rots
        result._linvel = std_vector_std_vector_Vector3d_to_ndarray(linvels)
        result._angvel = std_vector_std_vector_Vector3d_to_ndarray(angvels)
        result._linacc = std_vector_std_vector_Vector3d_to_ndarray(linaccs)
        result._angacc = std_vector_std_vector_Vector3d_to_ndarray(angaccs)

        result._f_local = std_vector_std_vector_Vector3d_to_ndarray(f_local)
        result._t_local = std_vector_std_vector_Vector3d_to_ndarray(t_local)

        result._com_pos = std_vector_Vector3d_to_ndarray(com_pos)
        result._com_linvel = std_vector_Vector3d_to_ndarray(com_linvel)
        result._com_ang_momentums = std_vector_Vector3d_to_ndarray(com_ang_momentums)
        
        return result

    def GetLocalLinearVelocity(self) -> np.ndarray:
        """
        return Linear Veolcity in np.ndarray with shape (body_count, 3)
        """
        cdef std_vector_Vector3d_ptr ptr
        InvDynGetLocalLinearVelocity(self.pointer, &ptr)
        return std_vector_Vector3d_ptr_to_ndarray(ptr)

    def GetLocalAngularVelocity(self) -> np.ndarray:
        """
        return AngularVelocity in np.ndarray with shape (body_count, 3)
        """
        cdef std_vector_Vector3d_ptr ptr
        InvDynGetLocalAngularVelocity(self.pointer, &ptr)
        return std_vector_Vector3d_ptr_to_ndarray(ptr)

    def GetLocalLinearAcceleration(self) -> np.ndarray:
        """
        return Local Linear Acceleration in np.ndarray with shape (body_count, 3)
        """
        cdef std_vector_Vector3d_ptr ptr
        InvDynGetLocalLinearAcceleration(self.pointer, &ptr)
        return std_vector_Vector3d_ptr_to_ndarray(ptr)

    def GetLocalAngularAcceleration(self) -> np.ndarray:
        """
        return Local Angular Acceleration in np.ndarray with shape (body_count, 3)
        """
        cdef std_vector_Vector3d_ptr ptr
        InvDynGetLocalAngularAcceleration(self.pointer, &ptr)
        return std_vector_Vector3d_ptr_to_ndarray(ptr)

    def GetLocalForce(self) -> np.ndarray:
        """
        param: None
        return Local Force in np.ndarray with shape (body_count, 3)
        """
        cdef std_vector_Vector3d_ptr ptr
        InvDynGetLocalForce(self.pointer, &ptr)
        return std_vector_Vector3d_ptr_to_ndarray(ptr)

    def GetLocalForce0(self) -> np.ndarray:
        """
        param: None
        return Local Force of body 0 in np.ndarray with shape (3,)
        """
        cdef std_vector_Vector3d_ptr ptr
        InvDynGetLocalForce(self.pointer, &ptr)
        cdef np.ndarray[np.float64_t, ndim=1] res = np.zeros(3)
        res[0] = ptr.getValue(0, 0)
        res[1] = ptr.getValue(0, 1)
        res[2] = ptr.getValue(0, 2)
        return res

    def GetLocalTorque(self) -> np.ndarray:
        """
        param: None
        return Local Torque in np.ndarray with shape (body_count, 3)
        """
        cdef std_vector_Vector3d_ptr ptr
        InvDynGetLocalTorque(self.pointer, &ptr)
        return std_vector_Vector3d_ptr_to_ndarray(ptr)

    def GetLocalTorque0(self) -> np.ndarray:
        cdef std_vector_Vector3d_ptr ptr
        InvDynGetLocalTorque(self.pointer, &ptr)
        cdef np.ndarray[np.float64_t, ndim=1] res = np.zeros(3)
        res[0] = ptr.getValue(0, 0)
        res[1] = ptr.getValue(0, 1)
        res[2] = ptr.getValue(0, 2)
        return res

    def GetJointRotation(self) -> np.ndarray:
        """
        return Joint Rotation in np.ndarray with shape (body_count, 3, 3)
        """
        cdef std_vector_Matrix3d_ptr ptr
        InvDynGetJointRotation(self.pointer, &ptr)
        return std_vector_Matrix3d_ptr_to_ndarray(ptr)

    def GetBodyOrientation(self) -> np.ndarray:
        """
        return Body Orientation in np.ndarray with shape (body_count, 3, 3)
        """
        cdef std_vector_Matrix3d_ptr ptr
        InvDynGetBodyOrientation(self.pointer, &ptr)
        return std_vector_Matrix3d_ptr_to_ndarray(ptr)

    def GetJointQIndex(self) -> np.ndarray:
        """
        return JointQIndex in np.ndarray[np.int32_t] with shape (body_count,)
        """
        cdef const std_vector[int] * out = &(self.pointer.GetJointQIndex())
        return std_vector_int_to_ndarray(out[0])

    def GetTotalDof(self) -> int:
        """
        return int
        """
        return InvDygGetTotalJointDof(self.pointer)

    def GetJointDof(self, int bid) -> int:
        """
        Param: int bid
        return JointDof
        """
        return InvDynGetJointDof(self.pointer, bid)

    def ReRoot(self, int bid):
        """
        Param: int bid
        return: None
        """
        InvDynReRoot(self.pointer, bid)

    def ReComputeR(self):
        """
        Param: empty
        return: None
        """
        InvDynReComputeR(self.pointer)

    def ClearReRootFlag(self):
        """
        Param: empty
        return: None
        """
        InvDynClearReRootFlag(self.pointer)

    def ComputeBodyPositions(self):
        """
        Param: empty
        return Body Positions in np.ndarray with shape (body_count, 3)
        """
        cdef std_vector_Vector3d a
        InvDynComputeBodyPositions(self.pointer, &a)
        cdef np.ndarray[np.float64_t, ndim=2] res = std_vector_Vector3d_to_ndarray(a)
        return res

    def ComputeComMomentums(self):
        """
        Param: empty

        return:
        mass: double
        com: np.ndarray in shape (3,)
        com_linvel: np.ndarray in shape (3,)
        com_angmom: np.ndarray in shape (3,)
        com_inertia: np.ndarray in shape (3, 3)
        """
        cdef double Mass
        cdef Eigen_Vector3d com, linVelocity, angMomentum
        cdef Eigen_Matrix3d inertia
        InvDynComputeComMomentums(self.pointer, &Mass,
                                  &com, &linVelocity, &angMomentum, &inertia)
        return Mass, eigen_vec3_to_ndarray(com), eigen_vec3_to_ndarray(linVelocity), eigen_vec3_to_ndarray(angMomentum), eigen_mat3_to_ndarray(inertia) 

    def ComputeLocalW_q(self):
        raise NotImplementedError()

    def GetCls(self) -> np.ndarray:
        """
        return in np.ndarray with shape (body_count, 3)
        """
        cdef std_vector_Vector3d_ptr ptr
        InvDynGetCls(self.pointer, & ptr)
        return std_vector_Vector3d_ptr_to_ndarray(ptr)

    def GetDCls(self) -> np.ndarray:
        """
        return in np.ndarray with shape (body_count, 3)
        """
        cdef std_vector_Vector3d_ptr ptr
        InvDynGetDCls(self.pointer, & ptr)
        return std_vector_Vector3d_ptr_to_ndarray(ptr)

    cpdef destroy_immediate(self):
        """
        """
        if self.pointer != NULL:
            DestroyInverseDynamics(self.pointer)

    def print_joint_axies(self):
        self.pointer.print_joint_axies()

"""
def divide_force(f_: np.ndarray,
                 tau_: np.ndarray,
                 joint_pos_: np.ndarray, 
                 mu_: np.ndarray,
                 r_max_: np.ndarray,
                 com_pos_: np.ndarray,
                 body_pos_: np.ndarray,
                 height_eps_: double,
                 f_clip_max_: double,  # <= 3mg?
                 clip_min_d_: double = 1e-3):
    #Param:
    #f_: Force on CoM. np.ndarray with shape (batch, 3)
    #tau_: torque of CoM. np.ndarray with shape (batch, 3)
    #joint_pos_: global position of each joint. np.ndarray with shape (batch, num joint, 3)
    #mu_: contact mu of each body. np.ndarray with shape (num bodies,)
    #r_max_: max length of each body. np.ndarray with shape (num bodies,)
    #com_pos_: CoM Position. np.ndarray with shape (batch, 3)
    #body_pos_: each body's global position. np.ndarray with shape (batch, num body, 3)
    #height_eps_: if joint's height < height_eps_, there is a contact between character and ground
    #clip_max: float, clip contact y force
    #clip_min_d_: float,

    assert f_.ndim == 2 and pos_.ndim == 3 and f_.shape[0] == pos_.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] f = np.ascontiguousarray(f_)  # (batch, 3)
    cdef np.ndarray[np.float64_t, ndim=2] tau = np.ascontiguousarray(tau_) # (batch, 3)
    cdef np.ndarray[np.float64_t, ndim=3] pos = np.ascontiguousarray(pos_)  # (batch, num joint, 3)
    cdef np.ndarray[np.float64_t, ndim=2] y = np.ascontiguousarray(pos_[:, :, 1]) # (batch, num joint)
    cdef np.ndarray[np.float64_t, ndim=1] mu = np.ascontiguousarray(mu_) # (num body, )
    cdef np.ndarray[np.float64_t, ndim=1] r_max = np.ascontiguousarray(r_max_)
    cdef np.ndarray[np.float64_t, ndim=2] com_pos = np.ascontiguousarray(com_pos_) # (batch, 3)
    cdef np.ndarray[np.float64_t, ndim=3] body_pos = np.ascontiguousarray(body_pos_)
    cdef double eps = height_eps_, clip_min_d = clip_min_d_, f_clip_max = f_clip_max_
    cdef np.ndarray[np.uint8_t, ndim=1] contact_flg = np.max(y, axis=1) < eps
    cdef np.ndarray[np.int32_t, ndim=1] contact_idx
    cdef np.ndarray[np.float64_t, ndim=1] contact_y, yi, f_ratio, f_xz_ratio, tau_max
    cdef np.ndarray[np.float64_t, ndim=2] f_each, tau_each, com_to_body, f_each_xz, f_each_xz_len, tau_each_len

    cdef int i = 0, j, size_0 = f.shape[0], size_1 = f.shape[1]
    cdef np.ndarray[np.float64_t, ndim=3] out_force = np.zeros((size_0, size_1, 3))  # (batch, num joint, 3)
    cdef np.ndarray[np.float64_t, ndim=3] out_torque = np.zeros((size_0, size_1, 3))
    for i in range(size_0):
        if not contact_flg[i]:  # if there is no contact, set result to zero
            continue
        yi = y[i]
        contact_idx = np.asarray(np.argwhere(yi < eps).reshape(-1), np.int32)
        contact_y = np.clip(yi[contact_idx], clip_min_d)  # height of each contact point
        # divide force on CoM to each contact joint
        f_ratio = 1.0 / contact_y
        f_ratio /= np.sum(f_ratio)  # shape = (num contact,)
        f_each = f_ratio[:, None] @ f[i, None, :]  # shape = (num contact, 3)
        # divide torque on CoM to each contact joint
        tau_each = f_ratio[:, None] @ tau[i, None, :]  # shape = (num contact, 3)
        com_to_body = com_pos[i, None, :] - body_pos[i, contact_idx, :]  # shape = (num contact, 3)
        tau_each -= com_to_body
        # add contact constraint
        f_each[:, 1] = np.clip(f_each[:, 1], 0) # contact force on y axis should >= 0
        # friction force at x, z axies should <= \mu F_y
        f_each_xz_len = np.linalg.norm(f_each[:, [0, 2]], axis=-1, keepdims=True)  # (num contact, 1)
        f_each[:, [0, 2]] /= f_each_xz_len
        f_each[:, [0, 2]] *= np.minimum(f_each_xz_len, mu[contact_idx, None] * f_each[:, 1, None])
        # length or torque should be less than r_max * \sqrt{1 + \mu^2} F_y
        tau_max = r_max[contact_idx] * np.sqrt(1 + mu[contact_idx] ** 2) * f_each[:, 1] 
        tau_each_len = np.linalg.norm(tau_each, axis=-1, keepdims=True) # (num contact, 1)
        tau_each /= tau_each_len
        tau_each *= np.minimum(tau_each_len, tau_max[:, None])
        # set to result
        out_force[i, contact_idx, :] = f_each
        out_torque[i, contact_idx, :] = tau_each

    return out_force, out_torque
"""






def test1_matrixxd_wrapper(np.ndarray a) -> np.ndarray:
    """
        Test functions are decleard in Utils/test/EigenWrapperTest.h

        void MatrixXd_Test1(Eigen::MatrixXd& a); // a += 2
        void MatrixXd_Test1_Wrapper(Eigen_MatrixXd * ptr)
        {
	        MatrixXd_Test1(ptr->data);
        }

        Eigen_MatrixXd is the wrapper class of Eigen::MatrixXd.
        It's decleared in EigenBindingWrapper.h

        The defination of Eigen_MatrixXd is as follows:

        class Eigen_MatrixXd {
        public:
	        Eigen::MatrixXd data;
	        Eigen_MatrixXd() {}
	        Eigen_MatrixXd(unsigned int rows, unsigned int cols) { this->resize(rows, cols); }
	        double getValue(unsigned int i, unsigned int j) const { return data(i, j); }
	        void setValue(unsigned int i, unsigned int j, double value) { data(i, j) = value; }
	        void resize(unsigned int rows, unsigned int cols) { this->data = MatrixXd(rows, cols); }
	        size_t rows() const { return this->data.rows(); }
	        size_t cols() const { return this->data.cols(); }
        };
    """

    assert a.ndim == 2
    cdef Eigen_MatrixXd mat = Eigen_MatrixXd()  # create empty wrapper
    EigenMatrixXdFromNumpy(a, mat)  # convert numpy.ndarray to Eigen::MatrixXd in wrapper class
    MatrixXd_Test1_Wrapper(&mat)  # function wrapper in C++

    cdef np.ndarray res = EigenMatrixXdToNumpy(mat) # convert Eigen::MatrixXd to numpy.ndarray
    return res




cdef class TorqueAdder:
    """
    param: parent_body_idx_: list[int]
    child_body_idx: list[int]

    """
    cdef TorqueAddHelper * ptr
    cdef size_t joint_cnt
    cdef size_t body_cnt
    def __cinit__(self, list parent_body_idx_, list child_body_idx_, int body_cnt_):
        cdef std_vector[int] parent_body_idx
        cdef std_vector[int] child_body_idx
        py_list_int_to_std_vector_int(parent_body_idx_, parent_body_idx)
        py_list_int_to_std_vector_int(child_body_idx_, child_body_idx)
        self.joint_cnt = child_body_idx.size()
        self.body_cnt = body_cnt_

        self.ptr = TorqueAddHelperCreate(parent_body_idx, child_body_idx, self.body_cnt)

    def __init__(self, list parent_body_idx, list child_body_idx, int body_cnt_):
        pass

    def __copy__(self):
        return self

    def __eq__(self, TorqueAdder other):
        return self.ptr == other.ptr

    def __dealloc__(self):
        self.destroy_immediate()

    def destroy_immediate(self):
        if self.ptr != NULL:
            TorqueAddHelperDelete(self.ptr)
            self.ptr = NULL

    @property
    def body_count(self) -> int:
        return self.body_cnt

    @property
    def joint_count(self) -> int:
        return self.joint_cnt

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_parent_body(self):
        cdef const std_vector[int] * res = self.ptr.GetParentBody()
        return std_vector_int_to_ndarray(res[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_child_body(self):
        cdef const std_vector[int] * res = self.ptr.GetChildBody()
        return std_vector_int_to_ndarray(res[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backward(self, np.ndarray[np.float64_t, ndim=2] prev_grad_):
        """
        param: gradient of body tau in shape (body count, 3)
        return: gradient of joint tau in shape (joint count, 3)
        """
        cdef np.ndarray[np.float64_t, ndim=2] prev_grad = np.ascontiguousarray(prev_grad_)
        assert prev_grad.shape[0] == self.body_cnt

        cdef np.ndarray[np.float64_t, ndim=2] grad = np.zeros((self.joint_cnt, 3))
        self.ptr.backward(<const double *>prev_grad.data , <double *> grad.data)

        return grad

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_torque(self, np.ndarray[np.float64_t, ndim=2] tau_):
        """
        param: joint tau in np.ndarray

        return: body_torque
        """
        cdef np.ndarray[np.float64_t, ndim=2] tau = np.ascontiguousarray(tau_)
        assert tau.shape[0] == self.joint_cnt
        cdef np.ndarray[np.float64_t, ndim=2] body_tau = np.zeros((self.body_cnt, 3))
        self.ptr.add_torque_forward(<double *> body_tau.data, <const double *> tau.data)

        return body_tau


@cython.boundscheck(False)
@cython.wraparound(False)
def simple_mix_quaternion(np.ndarray quat_input, np.ndarray weight_input = None):
    # https://forum.unity.com/threads/average-quaternions.86898/
    assert (weight_input is None or quat_input.shape[0] == weight_input.shape[0]) and quat_input.shape[1] == 4
    cdef np.ndarray[np.float64_t, ndim=2] quat_in = np.ascontiguousarray(quat_input)
    cdef np.ndarray[np.float64_t, ndim=1] weight
    cdef int use_weight = <int>(weight_input is not None)
    if use_weight > 0:
        weight = np.ascontiguousarray(weight_input)
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(4)
    cdef size_t i = 0, j, num = quat_in.shape[0]
    cdef double sum_val = 0.0, flag = 1.0
    for i in range(num):
        sum_val = 0.0
        for j in range(4):
            sum_val += result[j] * quat_in[i, j]
        flag = -1.0 + 2.0 * (sum_val >= 0)
        # print(sum_val, flag)
        if use_weight:
            flag = flag * weight[i]
        for j in range(4):
            result[j] += flag * quat_in[i, j]

    return result / np.linalg.norm(result)


@cython.boundscheck(False)
@cython.wraparound(False)
def mix_quat_by_slerp(np.ndarray quat_input):  # This method is not good at all.
    assert quat_input.shape[1] == 4
    cdef np.ndarray[np.float64_t, ndim=2] quat_in = np.ascontiguousarray(quat_input)
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(4)
    mix_quaternion(<double *> quat_in.data, quat_in.shape[0], <double *> result.data)
    return result


# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_inv_single_fast(np.ndarray q1_):
    cdef np.ndarray[np.float64_t, ndim=1] q = np.zeros(4)
    quat_inv_single(<const double *> q1_.data, <double* > q.data)
    return q

# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_inv_single_backward_fast(np.ndarray q, np.ndarray grad_in):
    cdef np.ndarray[np.float64_t, ndim=1] grad_out = np.zeros(4)
    quat_inv_backward_single(<const double *>(q.data), <const double *>(grad_in.data), <double *> grad_out.data)
    return grad_out


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_inv_fast(np.ndarray q1_):
    cdef size_t num_quat = q1_.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] q = np.zeros((num_quat, 4))
    quat_inv_impl(
        <const double *> q1_.data,
        <double *> q.data,
        num_quat
    )
    return q

# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_inv_backward_fast(np.ndarray q_, np.ndarray grad_in):
    cdef size_t num_quat = q_.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] grad_out = np.zeros((num_quat, 4))
    quat_inv_backward_impl(
        <const double *> q_.data,
        <const double *> grad_in.data,
        <double *> grad_out.data,
        num_quat
    )
    return grad_out


# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_multiply_forward_single(np.ndarray q1_, np.ndarray q2_):
    cdef np.ndarray[np.float64_t, ndim=1] q = np.zeros(4)
    quat_multiply_single(
        <const double * >q1_.data,
        <const double * >q2_.data,
        <double *> q.data
    )
    return q

# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_multiply_forward_fast(np.ndarray q1_, np.ndarray q2_):
    cdef size_t num_quat = q1_.shape[0]
    assert num_quat == q2_.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] q = np.zeros((num_quat, 4))
    quat_multiply_forward(
        <const double *> q1_.data,
        <const double *> q2_.data,
        <double *> q.data,
        num_quat
    )
    return q


# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_multiply_backward_fast(np.ndarray q1_, np.ndarray q2_, np.ndarray grad_q_):
    cdef size_t num_quat = q1_.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] grad_q1 = np.zeros((num_quat, 4))
    cdef np.ndarray[np.float64_t, ndim=2] grad_q2 = np.zeros((num_quat, 4))
    quat_multiply_backward(
        <const double *> q1_.data,
        <const double *> q2_.data,
        <const double *> grad_q_.data,
        <double *> grad_q1.data,
        <double *> grad_q2.data,
        num_quat
    )
    return grad_q1, grad_q2


# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_multiply_forward_one2many_fast(np.ndarray q1_, np.ndarray q2_):
    pass

# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_apply_single_fast(np.ndarray q, np.ndarray v):
    cdef np.ndarray[np.float64_t, ndim=1] o = np.zeros(3)
    quat_apply_single(<const double * > q.data, <const double *> v.data, <double *> o.data)
    return o


# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_apply_forward_fast(np.ndarray q, np.ndarray v):
    cdef size_t num_quat = q.shape[0]
    assert num_quat == v.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] o = np.zeros((num_quat, 3))
    quat_apply_forward(
        <const double *> q.data,
        <const double *> v.data,
        <double *> o.data,
        num_quat
    )
    return o


# Add by Yulong Zhang
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_apply_forward_one2many_fast(np.ndarray q, np.ndarray v):
    cdef size_t num_quat = q.shape[0]
    cdef size_t num_vector = v.shape[0]
    assert num_quat == 1
    cdef np.ndarray[np.float64_t, ndim=2] o = np.zeros((num_vector, 3))
    quat_apply_forward_one2many(
        <const double *> q.data,
        <const double *> v.data,
        <double *> o.data,
        num_vector
    )
    return o


# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_apply_single_backward_fast(np.ndarray q, np.ndarray v, np.ndarray o_grad):
    cdef np.ndarray[np.float64_t, ndim=1] q_grad = np.zeros(4)
    cdef np.ndarray[np.float64_t, ndim=1] v_grad = np.zeros(3)
    quat_apply_backward_single(
        <const double *> q.data,
        <const double *> v.data,
        <const double *> o_grad.data,
        <double *> q_grad.data,
        <double *> v_grad.data
    )
    return q_grad, v_grad

# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_apply_backward_fast(np.ndarray q, np.ndarray v, np.ndarray o_grad):
    cdef size_t num_quat = q.shape[0]
    assert num_quat == v.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] q_grad = np.zeros((num_quat, 4))
    cdef np.ndarray[np.float64_t, ndim=2] v_grad = np.zeros((num_quat, 3))
    quat_apply_backward(
        <const double *> q.data,
        <const double *> v.data,
        <const double *> o_grad.data,
        <double *> q_grad.data,
        <double *> v_grad.data,
        num_quat
    )

    return q_grad, v_grad


# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def flip_quat_by_w_fast(np.ndarray q):
    cdef np.ndarray[np.float64_t, ndim=2] q_out = np.zeros_like(q)
    cdef size_t num_quat = q_out.shape[0]
    flip_quat_by_w_forward_impl(
        <const double *> q.data,
        <double *> q_out.data,
        num_quat
    )
    return q_out


@cython.boundscheck(False)
@cython.wraparound(False)
def flip_quat_by_w_backward_fast(np.ndarray q, np.ndarray grad_q):
    cdef np.ndarray[np.float64_t, ndim=2] grad_out = np.zeros_like(grad_q)
    cdef size_t num_quat = grad_out.shape[0]
    flip_quat_by_w_backward_impl(
        <const double *> q.data,
        <const double *> grad_q.data,
        <double *> grad_out.data,
        num_quat
    )
    return grad_out


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_vec6d_single_fast(np.ndarray q):
    cdef np.ndarray[np.float64_t, ndim=2] vec6d = np.zeros((3, 2))
    quat_to_vec6d_single(<const double *> q.data, <double *> vec6d.data)
    return vec6d


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_matrix_single_fast(np.ndarray q):
    cdef np.ndarray[np.float64_t, ndim=2] mat = np.zeros((3, 3))
    quat_to_matrix_forward_single(<const double *> q.data, <double *> mat.data)
    return mat


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_vec6d_fast(np.ndarray q):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=3] result = np.zeros((num_quat, 3, 2))
    quat_to_vec6d_impl(<const double *> q.data, <double *> result.data, num_quat)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_matrix_fast(np.ndarray q):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=3] result = np.zeros((num_quat, 3, 3))
    quat_to_matrix_impl(
        <const double *> q.data,
        <double *> result.data,
        num_quat
    )
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_matrix_backward_fast(np.ndarray q, np.ndarray grad_in):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] grad_out = np.zeros_like(q)
    quat_to_matrix_backward(
        <const double *> q.data,
        <const double *> grad_in.data,
        <double *> grad_out.data,
        num_quat
    )
    return grad_out


# Add by Yulong Zhang
@cython.boundscheck(False)
@cython.wraparound(False)
def six_dim_mat_to_quat_fast(np.ndarray mat):
    cdef size_t num_quat = mat.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((num_quat, 4))
    six_dim_mat_to_quat_impl(
        <const double *> mat.data,
        <double *> result.data,
        num_quat
    )
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def vector_to_cross_matrix_fast(np.ndarray vec):
    cdef size_t num_vec = vec.shape[0]
    cdef np.ndarray[np.float64_t, ndim=3] mat = np.zeros((num_vec, 3, 3))
    vector_to_cross_matrix_impl(
        <const double *> vec.data,
        <double *> mat.data,
        num_vec
    )
    return mat


@cython.boundscheck(False)
@cython.wraparound(False)
def vector_to_cross_matrix_backward_fast(np.ndarray vec, np.ndarray grad_in):
    cdef size_t num_vec = vec.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] grad_out = np.zeros((num_vec, 3))
    vector_to_cross_matrix_backward(
        <const double *> vec.data,
        <const double *> grad_in.data,
        <double *> grad_out.data,
        num_vec
    )
    return grad_out


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_rotvec_single_fast(np.ndarray q):
    cdef double angle = 0.0
    cdef np.ndarray[np.float64_t, ndim=1] rotvec = np.zeros(3)
    quat_to_rotvec_single(
        <const double *> q.data,
        angle,
        <double *> rotvec.data
    )
    return rotvec


# here we also output the angle, for fast performance..
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_rotvec_fast(np.ndarray[np.float64_t, ndim=2] q):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] angle = np.zeros(num_quat)
    cdef np.ndarray[np.float64_t, ndim=2] rotvec = np.zeros((num_quat, 3))
    quat_to_rotvec_impl(
        <const double *> q.data,
        <double *> angle.data,
        <double *> rotvec.data,
        num_quat
    )
    return angle, rotvec


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_rotvec_fast2(np.ndarray[np.float64_t, ndim=2] q):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] angle = np.zeros(num_quat)
    cdef np.ndarray[np.float64_t, ndim=2] rotvec = np.zeros((num_quat, 3))
    quat_to_rotvec_impl(
        <const double *> q.data,
        <double *> angle.data,
        <double *> rotvec.data,
        num_quat
    )
    return rotvec


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_rotvec_backward_fast(
    np.ndarray[np.float64_t, ndim=2] q, np.ndarray[np.float64_t, ndim=1] angle, np.ndarray[np.float64_t, ndim=2] grad_in):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] grad_out = np.zeros((num_quat, 4))
    quat_to_rotvec_backward(
        <const double *> q.data,
        <const double *> angle.data,
        <const double *> grad_in.data,
        <double *> grad_out.data,
        num_quat
    )
    return grad_out


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_from_rotvec_single_fast(np.ndarray[np.float64_t, ndim=1] x):
    cdef np.ndarray[np.float64_t, ndim=1] q = np.zeros(4)
    quat_from_rotvec_single(<const double *> x.data, <double *> q.data)
    return q


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_from_rotvec_fast(np.ndarray[np.float64_t, ndim=2] x):
    cdef size_t num_quat = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] q = np.zeros((num_quat, 4))
    quat_from_rotvec_impl(<const double *> x.data, <double *> q.data, num_quat)
    return q


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_from_rotvec_backward_fast(np.ndarray[np.float64_t, ndim=2] x, np.ndarray[np.float64_t, ndim=2] grad_in):
    cdef size_t num_quat = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] grad_out = np.zeros((num_quat, 3))
    quat_from_rotvec_backward_impl(<const double *> x.data, <const double * > grad_in.data, <double *> grad_out.data, num_quat)
    return grad_out


# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_from_matrix_single_fast(np.ndarray[np.float64_t, ndim=2] mat):
    cdef np.ndarray[np.float64_t, ndim=1] q = np.zeros(4)
    quat_from_matrix_single(<const double *> mat.data, <double *> q.data)
    return q


# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_from_matrix_fast(np.ndarray[np.float64_t, ndim=2] mat):
    cdef size_t num_quat = mat.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] q = np.zeros((num_quat, 4))
    quat_from_matrix_impl(<const double *> mat.data, <double *> q.data, num_quat)
    return q


# Add by Zhenhua Song
@cython.boundscheck(False)
@cython.wraparound(False)
def quat_from_matrix_backward_fast(np.ndarray[np.float64_t, ndim=2] mat, np.ndarray[np.float64_t, ndim=2] grad_in):
    cdef size_t num_quat = mat.shape[0]
    cdef np.ndarray[np.float64_t, ndim=3] grad_out = np.zeros((num_quat, 3, 3))
    quat_from_matrix_backward_impl(<const double *> mat.data, <const double *> grad_in.data, <double *> grad_out.data, num_quat)
    return grad_out


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_hinge_angle_fast(np.ndarray[np.float64_t, ndim=1] q, np.ndarray[np.float64_t, ndim=2] axis):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] angle = np.zeros(num_quat)
    quat_to_hinge_angle_forward(
        <const double *> q.data,
        <const double *> axis.data,
        <double *> angle.data,
        num_quat
    )
    return angle


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_hinge_angle_backward_fast(np.ndarray[np.float64_t, ndim=1] q, np.ndarray[np.float64_t, ndim=1] axis, np.ndarray[np.float64_t, ndim=1] grad_in):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] grad_out = np.zeros_like(q)
    quat_to_hinge_angle_backward(
        <const double *> q.data,
        <const double *> axis.data,
        <const double *> grad_in.data,
        <double *> grad_out.data,
        num_quat
    )
    return grad_out


@cython.boundscheck(False)
@cython.wraparound(False)
def parent_child_quat_to_hinge_angle_fast(
    np.ndarray[np.float64_t, ndim=2] quat0,
    np.ndarray[np.float64_t, ndim=2] quat1,
    np.ndarray[np.float64_t, ndim=2] init_rel_quat_inv,
    np.ndarray[np.float64_t, ndim=2] axis
):
    cdef size_t num_quat = quat0.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] angle = np.zeros(num_quat)
    parent_child_quat_to_hinge_angle(
        <const double *> quat0.data,
        <const double *> quat1.data,
        <const double *> init_rel_quat_inv.data,
        <const double *> axis.data,
        <double *> angle.data,
        num_quat
    )
    return angle


@cython.boundscheck(False)
@cython.wraparound(False)
def parent_child_quat_to_hinge_angle_backward_fast(
    np.ndarray[np.float64_t, ndim=2] quat0, np.ndarray[np.float64_t, ndim=2] quat1,
    np.ndarray init_rel_quat_inv, np.ndarray axis, np.ndarray grad_in
):
    cdef size_t num_quat = quat0.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] quat0_grad = np.zeros_like(quat0)
    cdef np.ndarray[np.float64_t, ndim=2] quat1_grad = np.zeros_like(quat0)
    parent_child_quat_to_hinge_angle_backward(
        <const double *> quat0.data,
        <const double *> quat1.data,
        <const double *> init_rel_quat_inv.data,
        <const double *> axis.data,
        <const double *> grad_in.data,
        <double *> quat0_grad.data,
        <double *> quat1_grad.data,
        num_quat
    )
    return quat0_grad, quat1_grad


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_integrate_fast(np.ndarray q, np.ndarray omega, double dt):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros_like(q)
    quat_integrate_impl(
        <const double *> q.data,
        <const double *> omega.data,
        dt,
        <double *> result.data,
        num_quat
    )
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_integrate_backward_fast(np.ndarray q, np.ndarray omega, double dt, np.ndarray grad_in):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] q_grad = np.zeros((num_quat, 4))
    cdef np.ndarray[np.float64_t, ndim=2] omega_grad = np.zeros((num_quat, 3))
    quat_integrate_backward(
        <const double *> q.data,
        <const double *> omega.data,
        dt,
        <const double *> grad_in.data,
        <double *> q_grad.data,
        <double *> omega_grad.data,
        num_quat
    )
    return q_grad, omega_grad


@cython.boundscheck(False)
@cython.wraparound(False)
def vector_normalize_single_fast(np.ndarray x):
    cdef size_t ndim = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros_like(x)
    vector_normalize_single(
        <const double *> x.data,
        ndim,
        <double *> result.data
    )
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def vector_normalize_single_backward_fast(np.ndarray x, np.ndarray grad_in):
    cdef size_t ndim = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] grad_out = np.zeros_like(x)
    vector_normalize_backward_single(
        <const double *> x.data,
        ndim,
        <const double *> grad_in.data,
        <double *> grad_out.data
    )
    return grad_out


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_normalize_fast(np.ndarray q):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] q_out = q.copy()
    normalize_quaternion_impl(
        <const double *> q.data,
        <double *> q_out.data,
        num_quat
    )
    return q_out
    

# Add by Yulong Zhang
@cython.boundscheck(False)
@cython.wraparound(False)
def surface_distance_capsule(np.ndarray relative_distance, double radius, double length):
    cdef size_t ndim = relative_distance.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] surface_distance = np.zeros(ndim)
    cdef np.ndarray[np.float64_t, ndim=2] normal = np.zeros((ndim, 3))
    calc_surface_distance_to_capsule(
        <const double *> relative_distance.data,
        ndim,
        radius,
        length,
        <double *> surface_distance.data,
        <double *> normal.data
    )
    return surface_distance, normal

@cython.boundscheck(False)
@cython.wraparound(False)
def decompose_rotation_single_fast(np.ndarray q, np.ndarray vb):
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(4)
    decompose_rotation_single(
        <const double *> q.data,
        <const double *> vb.data,
        <double *> result.data
    )
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def facing_decompose_rotation_single_fast(np.ndarray q):
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(4)
    cdef double axis[3]
    axis[0] = 0
    axis[1] = 1
    axis[2] = 0
    decompose_rotation_single(
        <const double *> q.data,
        axis,
        <double *> result.data
    )
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def facing_decompose_rotation_inv_single_fast(np.ndarray q):
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(4)
    cdef double * result_ptr = <double *> result.data
    cdef double axis[3]
    axis[0] = 0
    axis[1] = 1
    axis[2] = 0
    decompose_rotation_single(
        <const double *> q.data,
        axis,
        result_ptr
    )
    quat_inv_single(result_ptr, result_ptr)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def decompose_rotation_fast(np.ndarray q, np.ndarray vb):
    cdef size_t num_quat = vb.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((num_quat, 4))
    decompose_rotation(
        <const double *> q.data,
        <const double *> vb.data,
        <double *> result.data,
        num_quat
    )
    # compute the result
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def decompose_rotation_single_pair_fast(np.ndarray q, np.ndarray vb):
    cdef np.ndarray[np.float64_t, ndim=1] qa = np.zeros(4)
    cdef np.ndarray[np.float64_t, ndim=1] qb = np.zeros(4)
    decompose_rotation_pair_single(
        <const double *> q.data,
        <const double *> vb.data,
        <double *> qa.data,
        <double *> qb.data
    )
    return qa, qb


@cython.boundscheck(False)
@cython.wraparound(False)
def decompose_rotation_pair_one2many_fast(np.ndarray q, np.ndarray vb):
    cdef size_t num_quat = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] qa = np.zeros((num_quat, 4))
    cdef np.ndarray[np.float64_t, ndim=2] qb = np.zeros((num_quat, 4))
    decompose_rotation_pair_one2many(
        <const double *> q.data,
        <const double *> vb.data,
        <double *> qa.data,
        <double *> qb.data,
        num_quat
    )
    return qa, qb


@cython.boundscheck(False)
@cython.wraparound(False)
def wxyz_to_xyzw_single(np.ndarray q):
    cdef np.ndarray[np.float64_t, ndim=1] res = np.empty(4)
    cdef double * data = <double *> q.data
    res[0] = data[1]
    res[1] = data[2]
    res[2] = data[3]
    res[3] = data[0]
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_cg_linear_solve(np.ndarray a_, np.ndarray b_):
    cdef size_t num = a_.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] a = np.ascontiguousarray(a_)
    cdef np.ndarray[np.float64_t, ndim=1] b = np.ascontiguousarray(b_)
    cdef np.ndarray[np.float64_t, ndim=1] x = np.zeros(num)

    eigen_solve_conjugate_gradient(
        <double *> a.data,
        <double *> b.data,
        <double *> x.data,
        num
    )

    return x


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_llt_linear_solve(np.ndarray a_, np.ndarray b_):
    cdef size_t num = a_.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] a = np.ascontiguousarray(a_)
    cdef np.ndarray[np.float64_t, ndim=1] b = np.ascontiguousarray(b_)
    cdef np.ndarray[np.float64_t, ndim=1] x = np.empty(num)
    eigen_solve_llt(
        <double *> a.data,
        <double *> b.data,
        <double *> x.data,
        num
    )
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_colPivHouseholderQr_linear_solve(np.ndarray a_, np.ndarray b_):
    cdef size_t num = a_.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] a = np.ascontiguousarray(a_)
    cdef np.ndarray[np.float64_t, ndim=1] b = np.ascontiguousarray(b_)
    cdef np.ndarray[np.float64_t, ndim=1] x = np.empty(num)
    eigen_solve_colPivHouseholderQr(
        <double *> a.data,
        <double *> b.data,
        <double *> x.data,
        num
    )
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_gae_fast(np.ndarray[np.float32_t, ndim=1] v_, np.ndarray[np.float32_t, ndim=1] r_, int term, float gamma_, float lamb_):
    cdef np.ndarray[np.float32_t, ndim=1] v = v_
    cdef np.ndarray[np.float32_t, ndim=1] r = r_
    cdef np.ndarray[np.float32_t, ndim=1] adv = np.zeros_like(r, np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] ret = r.copy()
    cdef int v_size = <int> v_.shape[0], a_size = <int> adv.shape[0], r_size = <int> r_.shape[0]
    cdef float gamma = gamma_
    cdef float lamb = lamb_
    cdef float delta
    cdef int step
    if term:
        adv[a_size - 1] = 0
    else:
        adv[a_size - 1] = r[r_size - 1] + gamma * v[v_size - 1] - v[v_size - 2]
        ret[r_size - 1] = v[r_size - 2] + gamma * v[v_size - 1]
    for step in range(r_size - 2, -1, -1):
        delta = r[step] + gamma * v[step + 1] - v[step]
        ret[step] = r[step] + gamma * ret[step + 1]
        adv[step] = delta + lamb * adv[step + 1]

    return adv, ret


    cdef AuxControl * handle

    def __cinit__(
        self,
        np.ndarray[np.float64_t, ndim=1] gravity_,
        np.ndarray[np.float64_t, ndim=1] body_mass_,
        np.ndarray[np.float64_t, ndim=3] body_init_inertia_,
        np.ndarray[np.float64_t, ndim=2] init_parent_joint_rel_pos_,
        np.ndarray[np.int32_t, ndim=1] parent_body_index_,
        int use_coriolis = 0,
        int use_gravity = 1
    ):
        # parent_body_index_ = parent_body_index_.astype(np.int32)
        self.handle = new AuxControl(
            <const double *> gravity_.data,
            <const double *> body_mass_.data,
            <const double *> body_init_inertia_.data,
            <const double *> init_parent_joint_rel_pos_.data,
            <const int *> parent_body_index_.data,
            body_mass_.size,
            use_coriolis,
            use_gravity
        )

    def __dealloc__(self):
        del self.handle

    def __init__(
        self,
        np.ndarray[np.float64_t, ndim=1] gravity_,
        np.ndarray[np.float64_t, ndim=1] body_mass_,
        np.ndarray[np.float64_t, ndim=3] body_init_inertia_,
        np.ndarray[np.float64_t, ndim=2] init_parent_joint_rel_pos_,
        np.ndarray[np.int32_t, ndim=1] parent_body_index_,
        int use_coriolis = 0,
        int use_gravity = 1
    ):
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def compute_body_torque_gravity(
        self,
        np.ndarray[np.float64_t, ndim=2] body_pos,
        np.ndarray[np.float64_t, ndim=3] body_rot_matrix
    ):
        cdef const double * result = self.handle.compute_body_torque(
            <const double * > body_pos.data,
            <const double * > body_rot_matrix.data,
            NULL
        )

        # Here we should not copy. instead, get the memory view
        return np.asarray(<np.float64_t[:self.handle.num_body, :3]> result)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def compute_body_torque(
        self,
        np.ndarray[np.float64_t] body_pos,
        np.ndarray[np.float64_t] body_rot_matrix,
        np.ndarray[np.float64_t] body_angvel
    ):
        cdef const double * result = self.handle.compute_body_torque(
            <const double * > body_pos.data,
            <const double * > body_rot_matrix.data,
            <const double * > body_angvel.data
        )

        # Here we should not copy. instead, get the memory view
        return np.asarray(<np.float64_t[:self.handle.num_body, :3]> result)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def gravity(self):
        return np.asarray(<np.float64_t[:3]> self.handle.gravity)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def init_parent_joint_rel_pos(self):
        return np.asarray(<np.float64_t[:self.handle.num_body, :3]> self.handle.init_parent_joint_rel_pos.data())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def tree_mass(self):
        return np.asarray(<np.float64_t[:self.handle.num_body]> self.handle.tree_mass.data())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def tree_com(self):
        return np.asarray(<np.float64_t[:self.handle.num_body, :3]> self.handle.tree_com.data())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def body_inertia(self):
        return np.asarray(<np.float64_t[:self.handle.num_body, :3, :3]> self.handle.body_inertia.data())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def coriolis(self):
        return np.asarray(<np.float64_t[:self.handle.num_body, :3]> self.handle.coriolis.data())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def joint_position(self):
        return np.asarray(<np.float64_t[:self.handle.num_body, :3]> self.handle.joint_position.data())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def body_torque_result(self):
        return np.asarray(<np.float64_t[:self.handle.num_body, :3]> self.handle.body_torque_result.data())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def tree_coriolis(self):
        return np.asarray(<np.float64_t[:self.handle.num_body, :3]> self.handle.tree_coriolis.data())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def end_body_index(self):
        cdef int num_end = self.handle.end_body_index.size()
        cdef np.ndarray[np.int32_t, ndim=1] result = np.empty(num_end, np.int32)
        cdef int i
        for i in range(num_end):
            result[i] = self.handle.end_body_index[i]
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def is_end_body(self):
        cdef int num_body = <int> self.handle.num_body
        cdef np.ndarray[np.int32_t, ndim=1] result = np.empty(num_body, np.int32)
        cdef int i
        for i in range(num_body):
            result[i] = self.handle.is_end_body[i]
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def index_queue(self):
        cdef int num_body = self.handle.num_body
        cdef np.ndarray[np.int32_t, ndim=1] result = np.empty(num_body, np.int32)
        cdef int i
        for i in range(num_body):
            result[i] = self.handle.index_queue[i]
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def use_gravity(self):
        return self.handle.use_gravity

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @use_gravity.setter
    def use_gravity(self, value):
        self.handle.use_gravity = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @property
    def use_coriolis(self):
        return self.handle.use_coriolis

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @use_coriolis.setter
    def use_coriolis(self, value):
        self.handle.use_coriolis = value




def delta_quat_float32(
    np.ndarray[np.float32_t, ndim=2] prev_quat,
    np.ndarray[np.float32_t, ndim=2] quat,
    float dt
):
    cdef int batch = <int>quat.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] res = np.empty((batch, 3), np.float32)
    delta_quat2_impl_float32(
        <const float *>prev_quat.data,
        <const float *>quat.data,
        <float> (1 / dt),
        <float *> res.data,
        batch
    )
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def quat_between_single_fast(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(4)
    quat_between_single(<const double*> a.data, <const double*> b.data, <double*> result.data)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def resize_image(np.ndarray[np.float32_t, ndim=2] pose_2d_):
    cdef np.ndarray[np.float32_t, ndim=3] pose_2d = pose_2d_.reshape((pose_2d_.shape[0], -1, 2))
    cdef np.ndarray[np.float32_t, ndim=3] pose_min = np.min(pose_2d, -2, None, True) # [x_min, y_min]
    cdef np.ndarray[np.float32_t, ndim=3] pose_max = np.max(pose_2d, -2, None, True)  # [x_max, y_max]
    cdef np.ndarray[np.float32_t, ndim=3] pose_width = pose_max - pose_min  # [x_max - x_min, y_max - y_min]
    cdef np.ndarray[np.float32_t, ndim=3] pose_center = 0.5 * (pose_min + pose_max)
    cdef np.ndarray[np.float32_t, ndim=3] pose_scale = 1.0 / (pose_width + 1e-7)
    cdef np.ndarray[np.float32_t, ndim=3] result = (pose_2d - pose_center) * pose_scale
    return pose_2d