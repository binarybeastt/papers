import numpy as np
import scipy
class DotProductAttention:
    def __init__(self, query, key, mask, value, scale=True):
        self.query = query
        self.key = key
        self.mask = mask
        self.value = value
        self.scale = scale
        
    def dp_attention(self):
        """Dot product self-attention.
        Args:
            query (numpy.ndarray): array of query representations with shape (L_q by d)
            key (numpy.ndarray): array of key representations with shape (L_k by d)
            value (numpy.ndarray): array of value representations with shape (L_k by d) where L_v = L_k
            mask (numpy.ndarray): attention-mask, gates attention with shape (L_q by L_k)
            scale (bool): whether to scale the dot product of the query and transposed key

        Returns:
            numpy.ndarray: Self-attention array for q, k, v arrays. (L_q by L_k)
        """
        assert self.query.shape[-1] == self.key.shape[-1] == self.value.shape[-1]
        if self.scale:
            depth = self.query.shape[-1]
        else:
            depth = 1
            
        dots = np.matmul(self.query, np.swapaxes(self.key, -1, -2))/np.sqrt(depth)
        if self.mask is not None:
            dots = np.where(self.mask, dots, np.full_like(dots, -1e9))
            
        logsumexp = scipy.special.logsumexp(dots, axis=1, keepdims=True)
        dots = np.exp(dots - logsumexp)

        # Multiply dots by value to get self-attention
        # Use np.matmul()
        attention = np.matmul(dots, self.value)
        
        return attention
    
    # def dot_product_self_attention(self):
    #     """ Masked dot product self attention.
    #     Args:
    #         q (numpy.ndarray): queries.
    #         k (numpy.ndarray): keys.
    #         v (numpy.ndarray): values.
    #     Returns:
    #         numpy.ndarray: masked dot product self attention tensor.
    #     """
        
    #     # Size of the penultimate dimension of the query
    #     mask_size = self.query.shape[-2]

    #     # Creates a matrix with ones below the diagonal and 0s above. It should have shape (1, mask_size, mask_size)
    #     # Use np.tril() - Lower triangle of an array and np.ones()
    #     mask = np.tril(np.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)  
            
    #     return self.dp_attention(self)
    

def create_tensor(t):
        return np.array(t)
    
def display_tensor(t, name):
        """Display shape and tensor"""
        print(f'{name} shape: {t.shape}\n')
        print(f'{t}\n')
        
query=create_tensor([[1,2,3], [4,5,6]])
key=create_tensor([[2,9,4], [3,4,6]])
value=create_tensor([[8,7,6], [4,9,10]])

attention_s = DotProductAttention(query=query, key=key, mask=None, value=value)
print(attention_s.dp_attention())
