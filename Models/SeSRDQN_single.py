# -*- coding: utf-8 -*-
from turtle import distance, shape
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np
from Utils.Utils import sparse_to_dense

class SeSRDQN_single(object):
    def __init__(self, n_s, lambda_d = 0.01, lambda_div = 0.001, lambda_enc = 0.001, lambda_freq = 0.01, verbose =1,learning_rate = 0.1, embsize =16, pool_size =100, salary_encoder_layers=[24, 16], salary_decoder_layers = [24, 16], dropout_salary = [1.0, 1.0, 1.0], salary_prototype = 21, dropout_prototype = [1.0, 1.0, 1.0], activation = tf.nn.relu, random_seed = 2022, l2_reg = 0.0, name="", sess=None):
        
        self.n_s = n_s
        self.lambda_d = lambda_d
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.embsize = embsize
        self.pool_size = pool_size
        self.lambda_div = lambda_div
        self.lambda_enc = lambda_enc
        self.lambda_freq = lambda_freq

        self.salary_encoder_layers = salary_encoder_layers
        self.salary_decoder_layers = salary_decoder_layers
        self.dropout_salary_feed = dropout_salary
        self.salary_prototype = salary_prototype
        self.dropout_prototype_feed = dropout_prototype

        self.activation = activation
        self.random_seed = random_seed
        self.l2_reg = l2_reg
        self.name = name 
        self.sess = sess

        self.weights = {}
        # self.attention = {}
        self.dropouts = {}
        self._init_graph()

    def _init_graph(self):
        tf.set_random_seed(self.random_seed)
        self._build_network()
        #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        # init 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.saver = tf.train.Saver(max_to_keep=1)
        init = tf.global_variables_initializer()
        if self.sess is None:
            self.sess = tf.Session(config=config)
            self.sess.run(init)

        print("#params: %d" % self._count_parameters())

    def run(self, data, batch_size, train=True):
        n_data = len(data[0])
        predictions, loss = [], []
        for i in range(0, n_data, batch_size):
            # data_batch = [dt[i:min(i + batch_size, n_data)] for dt in data]
            data_batch = data
            if train:
                preds, loss_step, _ = self.sess.run(
                    (self.v, self.loss, self.optimizer),
                    feed_dict=self.get_dict(data_batch, train=True))

            else:
                # retrieve the corresponding data run( , )
                preds = self.sess.run((self.v), feed_dict=self.get_dict(data_batch, train=False))
            predictions.append(preds)
            loss.append(loss_step)

        return predictions, loss

    def get_dict(self, data, train=True):
        feed_dict = {
            self.input_state: data[0],
            self.input_action: data[1],
            self.label: data[2],
            self.frequency_set: data[3],
            self.salary_input_state_emb_dropout: self.dropout_prototype_feed if train else [1] * len(self.dropout_prototype_feed),
            # self.deep_encoder_salary_action_dropout: self.dropout_salary_feed if train else [1] * len(self.dropout_salary_feed),
            self.deep_decoder_salary_dropout: self.dropout_salary_feed if train else [1] * len(self.dropout_salary_feed),
        }
        return feed_dict

    def _count_parameters(self, print_count=False):
        total_parameters = 0
        for name, variable in self.weights.items():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            if print_count:
                print(name, variable_parameters)
            total_parameters += variable_parameters
        return total_parameters
    
    def _build_network(self):
        # --------------------------input ------------------------
        self.input_state = tf.placeholder(tf.float32, shape=[None, self.n_s], name = "%s_input_state" % self.name) # -1, n_s
        self.label = tf.placeholder(tf.float32, shape = [None], name = "%s_input_label" % self.name)
        self.input_action = tf.placeholder(tf.int32, shape=[None, self.pool_size], name="%s_input_action" % self.name) # num, pool_size
        self.frequency_set = tf.placeholder(tf.float32, shape=[None, self.n_s], name = "%s_frequency_set" % self.name)# num * 10, n_s


        # ----------------------- multi-hot processing ----------------------
        self.batch_size = tf.shape(self.input_state)[0]
        action_set = tf.one_hot(self.input_action, self.n_s)

        # ----------------------- salary skill embeddings -------------------------
        if "%s_salary_embedding" % self.name not in self.weights:
            self.salary_emb = tf.Variable(np.random.normal(size=(self.n_s, self.embsize)),
                          dtype=np.float32, name="%s_salary_embedding" % self.name)
        self.weights["%s_salary_embedding" % self.name] = self.salary_emb

        # ----------------------- salary action weights -----------------------
        if "%s_salary_skill_action_value" % self.name not in self.weights:
            self.salary_skill_action_value = tf.Variable(np.random.normal(size=(self.n_s, self.salary_prototype)),
                          dtype=np.float32, name="%s_salary_skill_action_value" % self.name)
        self.weights["%s_salary_skill_action_value" % self.name] = self.salary_skill_action_value


        # ---------------------- encoder network -------------------------
        # self.deep_salary_dropout, y_deep_encoder, _ = self._mlp(mix_input, self.n_s, self.salary_encoder_layers,
        #                                                name="%s_deep_encoder_salary" % self.name,
        #                                                activation=self.activation, bias=True, sparse_input=True, batch_norm=False)
        
        input_state_expanded = tf.expand_dims(self.input_state, -1) # batch_size, n_s, 1
        salary_emb_expanded = tf.expand_dims(self.salary_emb, 0) # 1, n_s, embsize
        
        input_salary_emb = tf.reduce_sum(tf.multiply(input_state_expanded, salary_emb_expanded), axis=1)


        self.input_salary_emb = tf.reshape(input_salary_emb, [-1, 1, self.embsize]) # batch_size, 1, embsize
        self.raw_input_salary_emb = tf.reshape(input_salary_emb, [-1, 1, self.embsize]) # batch_size, 1, embsize

        self.freq_salary_emb  = tf.matmul(self.frequency_set, self.salary_emb)
        self.action_salary_emb = tf.matmul(action_set, self.salary_emb) 

        # ---------------------- decoder network ------------------
        # This is designed for the frequency set only, for prototype visualization
        self.deep_decoder_salary_dropout, y_salary_deep_decoder, _ = self._mlp(self.freq_salary_emb, self.embsize, self.salary_decoder_layers,
                                                        name="%s_deep_decoder_salary" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

         
        self.salary_decoder, _ = self._fc(y_salary_deep_decoder, self.salary_decoder_layers[-1], self.n_s, name="%s_salary_decoder_output" % self.name, l2_reg=0.0, activation= tf.nn.sigmoid, bias=True, sparse=False)

        # ---------------------- decoder network loss function ------------------
        # salary_l1_norm = tf.norm(self.salary_decoder, ord=1)
        # bce = tf.keras.losses.BinaryCrossentropy()
        # self.encoder_loss = bce(self.frequency_set, self.salary_decoder) 
        self.encoder_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(self.frequency_set, self.salary_decoder))

        if "%s_salary_prototype" % self.name not in self.weights:
            self.weights["%s_salary_prototype" % self.name] =  tf.Variable(tf.random_uniform(shape=[self.salary_prototype, self.embsize], dtype=tf.float32), name = "%s_salary_prototype" % self.name) # prototype_num, emdsize

        # ---------------------- action network ------------------ 
        #self.deep_encoder_salary_action_dropout, y_deep_action, _ = self._mlp(self.action_salary_emb, self.embsize, [self.embsize, self.embsize],
        #                                                name="%s_deep_encoder_salary_action" % self.name,
        #                                                activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        # For last layer, we use SoftPlus function
        #self.salary_action_value, _ = self._fc(y_deep_action, self.salary_encoder_layers[-1], self.salary_prototype, name="%s_salary_encoder_action" % self.name, l2_reg=0.0, activation = tf.nn.softplus, bias=True, sparse=False, batch_norm=False) # batch_size, pool_size, prototype_num
        
        # get the salary action value in the batch size and pool size
        # batch_size, pool_size, prototype_num
        # action_set: batch_size, pool_size, n_s
        # salary_skill_action_value: n_s, prototype_num
        self.salary_skill_action_value_softplus = tf.nn.softplus(self.salary_skill_action_value)
        self.salary_action_value = tf.matmul(action_set, self.salary_skill_action_value_softplus) # batch_size, pool_size, prototype_num
        

        # ---------------------- salary prototype ------------------
        # apply the same transformation layer on the each part
        self.salary_input_state_emb_dropout, self.input_salary_emb, _ = self._mlp(self.input_salary_emb, self.embsize, [self.embsize], name="%s_salary_input_state_emb" % self.name, activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        self.input_salary_emb, _ = self._fc(self.input_salary_emb, self.embsize, self.embsize, name="%s_salary_fc_input_state_emb" % self.name, l2_reg=0.0, activation=None, bias=True, sparse=False)


        self.q_salary, salary_diversity_loss, salary_freq_loss, _, salary_weights, self.skill_salary_attention = self._prototype_layer(self.input_salary_emb, self.freq_salary_emb, self.salary_prototype, self.embsize, self.salary_action_value, salary_flag=True, name="%s_salary" % self.name)
        self.q_salary = tf.reshape(self.q_salary, [-1, self.pool_size])
        self.frequency_loss = salary_freq_loss

        ## add l2 loss over the dim(-1) of self.salary_action_value
        salary_info_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.salary_action_value), axis=-1))
        # salary_info_loss = tf.nn.l2_loss(salary_weights)
        self.info_loss = salary_info_loss

        
        # ---------------------- mapping prototypes back ------------------
        _ , y_deep_prototype, _ = self._mlp(salary_weights, self.embsize, self.salary_decoder_layers,
                                                        name="%s_deep_decoder_salary" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        self.skill_salary_prototypes, _ = self._fc(y_deep_prototype, self.salary_decoder_layers[-1], self.n_s, name="%s_salary_decoder_output" % self.name, l2_reg=0.0, activation= tf.nn.sigmoid, bias=True, sparse=False) # prototype_num * n_s
        
        # ---------------------------------------------------------------------
        # --------------------- Choose an action ------------------------------
        # ---------------------------------------------------------------------
        self.q = self.q_salary  # -1, pool_size    

        self.v_place = tf.argmax(self.q, 1)

        onehot_action_nxt = tf.one_hot(self.v_place, self.pool_size)  # should be: data batch, pool_size  

        # ----------- has a problem -----------
        self.v_skill = tf.cast(tf.reduce_sum(tf.multiply(onehot_action_nxt, tf.cast(self.input_action, np.float32)), 1), np.int32)        
        self.v = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q), 1)
        self.v_salary = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q_salary), 1)

        # ------ attention selections-----------------------
        self.skill_salary_attention = tf.reshape(self.skill_salary_attention, [-1, self.pool_size, self.salary_prototype])

        att_onehot_salary = tf.tile(tf.reshape(onehot_action_nxt, [-1, self.pool_size, 1]), [1,1, self.salary_prototype])


        self.skill_salary_attention_final =tf.reduce_sum(tf.multiply(att_onehot_salary, self.skill_salary_attention), axis=1)

        # --------------------- loss ----------------------------------
        # --------------------- need to add lambda_d to restrict ------
        self.diversity_loss = tf.reduce_mean(salary_diversity_loss)

        self.loss = tf.reduce_mean(tf.square(self.v_salary - self.label))

        self.loss += self.lambda_div * self.diversity_loss
        self.loss += self.lambda_enc * self.encoder_loss
        self.loss += self.lambda_freq * self.frequency_loss
        return 

    def _similarity_function(self, tensor1, tensor2):
        tensor1_norm = tf.nn.l2_normalize(tensor1, axis=-1)
        tensor2_norm = tf.nn.l2_normalize(tensor2, axis=-1)

        # cosine similarity
        similarity = tf.reduce_sum(tf.multiply(tensor1_norm, tensor2_norm), axis=-1)
        return (similarity + 1)/2

    def _l2_distance_function(self, tensor1, tensor2):
        # euclidean distance
        distance = tf.norm(tf.cast(tensor1 - tensor2, tf.float32), ord='euclidean', axis=-1)
        return distance

    def _diversity_function(self, prototypes):
        # Ensure the prototypes tensor is of rank 3
        prototypes = tf.ensure_shape(prototypes, [1, None, None])

        # Normalize the prototypes along the embedding size dimension
        normalized_prototypes = tf.nn.l2_normalize(prototypes, -1)

        # Compute cosine similarity matrix
        cosine_similarity_matrix = tf.matmul(normalized_prototypes, normalized_prototypes, transpose_b=True)

        # Sum over upper triangle of the matrix, excluding the diagonal
        # The diagonal represents self-similarity which is always 1, so it should be excluded
        sum_upper_triangle = tf.reduce_sum(tf.matrix_band_part(cosine_similarity_matrix, 0, -1)) - tf.reduce_sum(tf.linalg.diag_part(cosine_similarity_matrix))

        # Count the number of elements in the upper triangle, excluding the diagonal
        prototype_num = tf.shape(prototypes)[1]
        num_elements = (prototype_num * (prototype_num - 1)) / 2

        # Calculate the mean
        mean_similarity = sum_upper_triangle / tf.cast(num_elements, tf.float32)

        return mean_similarity
    
    # contain the prototype duplicates
    def _prototype_layer(self, tensor, frequency_set, prototype_size, embedding_size, q_value, salary_flag, name):
        ### q_value: batch_size, pool_size, prototype_num
        ### tensor(input state): batch_size, 1, embedding_size
        ### frequency_set: -1, embedding_size

        #if "%s_prototype" % name not in self.weights:
        #    self.weights["%s_prototype" % name] =  tf.Variable(tf.random_uniform(shape=[prototype_size, embedding_size], dtype=tf.float32), name = #"%s_prototype" % name) # prototype_num, emdsize
        
        # apply the same transformation layer on the each part
        if salary_flag == True:
            _, self.tranformed_salary_prototypes, _ = self._mlp(self.weights["%s_prototype" % name], embedding_size, [embedding_size], name="%s_input_state_emb" % name, activation=self.activation, bias=True, sparse_input=False, batch_norm=False)
            self.tranformed_salary_prototypes, _ = self._fc(self.tranformed_salary_prototypes, embedding_size, embedding_size, name="%s_fc_input_state_emb" % name, l2_reg=0.0, activation=None, bias=True, sparse=False)
            prototypes = self.tranformed_salary_prototypes
        elif salary_flag == False:
            _, self.tranformed_difficulty_prototypes, _ = self._mlp(self.weights["%s_prototype" % name], embedding_size, [embedding_size], name="%s_input_state_emb" % name, activation=self.activation, bias=True, sparse_input=False, batch_norm=False)
            self.tranformed_difficulty_prototypes, _ = self._fc(self.tranformed_difficulty_prototypes, embedding_size, embedding_size, name="%s_fc_input_state_emb" % name, l2_reg=0.0, activation=None, bias=True, sparse=False)  
            prototypes = self.tranformed_difficulty_prototypes

        # prototypes, _ = self._fc(deep_prototypes, embedding_size, embedding_size, name="%s_difficulty_encoder_action" % name, l2_reg=0.0, activation = tf.nn.relu, bias=True, sparse=False)

        # tensor_prot = tf.broadcast_to(tf.reshape(tensor, [-1, 1, embedding_size]), [self.batch_size * self.pool_size, prototype_size, self.embsize]) # num * pool_size, prototype_num, emdsize

        # tensor_prot = tf.tile(tf.reshape(tensor, [-1, 1, embedding_size]), [1, prototype_size, 1])
        tensor_prot = tf.expand_dims(tensor, 2) # batch_size, 1, 1, embedding_size
        prototypes = tf.expand_dims(prototypes, 0) # 1, prototype_num, embedding_size
        # print(tensor_prot.get_shape(), prototypes.get_shape())

        similarity = self._similarity_function(tensor_prot, tf.expand_dims(prototypes, 0)) # batch_size, 1, prototype_num
        # cancel softmax
        # similarity = tf.nn.softmax(similarity, axis=-1) # batch_size, 1, prototype_num

        #self.attention["%s_prototype_weights" % name] = tf.nn.softmax(similarity, axis=1) # -1, prototype_num
        # attention = tf.nn.softmax(similarity, axis=1) # -1, prototype_num
        
        # debug unit
        # check0 = tf.debugging.assert_equal(similarity, q_value)
        q_result = tf.reduce_sum(tf.multiply(similarity, q_value), axis=-1, keepdims=True) # batch_size, pool_size, prototype_num
        
        # clusering and evidence loss
        frequency_loss = 0
        
        # freq_prot = tf.broadcast_to(tf.reshape(frequency_set, [-1, 1, embedding_size]), [self.batch_size * 4, prototype_size, self.embsize]) # num * sampling size, prototype_num, emdsize
        
        # add transformation layer
        # freq_prot = frequency_set
        _, freq_prot, _ = self._mlp(frequency_set, embedding_size, [embedding_size], name="%s_input_state_emb" % name, activation=self.activation, bias=True, sparse_input=False, batch_norm=False)
        freq_prot, _ = self._fc(freq_prot, embedding_size, embedding_size, name="%s_fc_input_state_emb" % name, l2_reg=0.0, activation=None, bias=True, sparse=False)

        freq_prot = tf.reshape(freq_prot, [-1, 1, embedding_size]) # num * sampling size, 1, emdsize

        freq_distance = self._l2_distance_function(freq_prot, prototypes) # -1, prototype_num

        # for each frequency set 
        freq_min_loss = tf.reduce_mean(tf.reduce_min(freq_distance, axis=1))
        frequency_loss += freq_min_loss

        # for each prototype
        evidence_prototype_loss = tf.reduce_mean(tf.reduce_min(freq_distance, axis=0))
        frequency_loss += evidence_prototype_loss
        
        # infoNCE loss 
        # for each frequency set
        # infoNCE_loss = tf.math.log(tf.reduce_min(freq_distance, axis=1)/tf.reduce_sum(freq_distance, axis=1))
        infoNCE_loss = tf.convert_to_tensor(0, dtype=tf.float32)

        # diversity loss
        diversity_loss = self._diversity_function(prototypes)

        return q_result, diversity_loss, frequency_loss, infoNCE_loss, self.weights["%s_prototype" % name], similarity



    def _fc(self, tensor, dim_in, dim_out, name, l2_reg, activation=None, bias=True, sparse=False):
        glorot = np.sqrt(2.0 / (dim_in + dim_out))
        # add if condition, if not existing 
        if "%s_w" % name not in self.weights:
            self.weights["%s_w" % name] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(dim_in, dim_out)),
                                                  dtype=np.float32, name="%s_w" % name)
        if not sparse:
            y_deep = tf.matmul(tensor, self.weights["%s_w" % name])
        else:
            y_deep = tf.sparse_tensor_dense_matmul(tensor, self.weights["%s_w" % name])
        if bias:
            if "%s_b" % name not in self.weights:
                self.weights["%s_b" % name] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, dim_out)),
                                                          dtype=np.float32, name="%s_b" % name)
                y_deep += self.weights["%s_b" % name]
        if activation is not None:
            y_deep = activation(y_deep)
        return y_deep, tf.contrib.layers.l2_regularizer(l2_reg)(self.weights["%s_w" % name])

    def _mlp(self, tensor, dim_in, layers, name, activation=None, bias=True, sparse_input=False, batch_norm=False):
        if name not in self.dropouts:
            self.dropouts[name] = tf.placeholder(tf.float32, shape=[None], name="%s_dropout" % name)
        dropout = self.dropouts[name]
        #y_deep = tf.nn.dropout(tensor, rate = 1.0 - dropout[0])
        y_deep = tensor
        lst = []
        loss = 0
        for i, layer in enumerate(layers):
            if i == 0 and sparse_input:
                y_deep, _ = self._fc(y_deep, dim_in, layer, l2_reg=self.l2_reg, name="%s_%d" % (name, i),
                                            bias=bias, activation=activation, sparse=True)
            else:
                y_deep, _ = self._fc(y_deep, dim_in, layer, l2_reg=self.l2_reg, name="%s_%d" % (name, i),
                                            bias=bias, activation=activation, sparse=False)
            # add NB
            #if batch_norm:
            # y_deep = tf.nn.leaky_relu(tf.layers.batch_normalization(y_deep, training=self.train))
            y_deep = tf.nn.dropout(y_deep, rate = 1.0 - dropout[i + 1])
            lst.append(y_deep)
            dim_in = layer
        return dropout, y_deep, loss

    def save(self, save_path):
        self.saver.save(self.sess, save_path)

    def load(self, load_path):
        self.saver.restore(self.sess, load_path)

    def evaluate_metrics(self, y_pred_salary, y_true_salary, y_pred_easy, y_true_easy):
        salary_mse = mean_squared_error(y_true_salary, y_pred_salary)
        easy_mse = mean_squared_error(y_true_easy, y_pred_easy)
        return [("salary_mse", salary_mse), ("easy_mse", easy_mse)]

    def predict(self, data, batch_size=32):
        predictions_salary, predictions_easy = self.run(data, batch_size, train=False)
        return predictions_salary, predictions_easy

    def print_result(self, data_eval, endch="\n"):
        print_str = ""
        for i, name_val in enumerate(data_eval):
            if i != 0:
                print_str += ','
            print_str += "%s: %f" % name_val
        print(print_str, end=endch)

    def estimate_maxq_action(self, state_vis, act_pool):
        data = [[state_vis], [act_pool], [0], [[0] * self.n_s]]
        act, v_salary = self.sess.run((self.v_skill, self.v_salary), self.get_dict(data, train=False))
        return v_salary[0], act[0]

    def estimate_maxq_batch(self, data_state, data_pool):
        n_data = len(data_state)
        act_lst, v_salary = self.sess.run((self.v_skill, self.v_salary),
                                                  self.get_dict((data_state, data_pool, [0] * n_data, [[0] * self.n_s] * n_data * 10), train=False))
        return v_salary, act_lst


    def evaluate_internal_value_functions(self, state_vis, act_pool):
        data = [[state_vis], [act_pool], [0], [sparse_to_dense([416, 222, 183, 743, 746], self.n_s)]]
        loss, encoder_loss, div_loss, freq_loss = self.sess.run((self.loss, self.encoder_loss, self.diversity_loss, self.frequency_loss), self.get_dict(data, train=False))
        return loss, encoder_loss, div_loss, freq_loss
    
    def get_q_list(self, skill_vis, act_pool):
        data = [[skill_vis], [act_pool], [0], [[0] * self.n_s]]
        q_salary_lst = self.sess.run((self.q_salary), self.get_dict(data, train=False))
        return q_salary_lst[0]

    # 
    def get_embeddings(self, state_vis, flag):
        input_state = [sparse_to_dense(state_vis, self.n_s)]
        data = [input_state, [[0] * self.pool_size], [0], [[0] * self.n_s]]
        if flag: 
           embeddings = self.sess.run((self.salary_embedding), self.get_dict(data, train=False))
        return embeddings[0]

    def get_salary_prototypes(self):
        data = [[[0] * self.n_s], [[0] * self.pool_size], [0], [[0] * self.n_s]]
        decoder_result, embeddings = self.sess.run((self.skill_salary_prototypes, self.weights["%s_salary_protoype" % self.name]), self.get_dict(data, train=False))
        return decoder_result, embeddings