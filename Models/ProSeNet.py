# -*- coding: utf-8 -*-
from turtle import distance, shape
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np
from Utils.Utils import sparse_to_dense

class ProSeNet(object):
    def __init__(self, n_s, lambda_d = 0.01, lambda_div = 0.001, lambda_info = 0.1, lambda_enc = 0.001, lambda_freq = 0.01, verbose =1,learning_rate = 0.1, embsize =16, pool_size =100, salary_encoder_layers=[24, 16], salary_decoder_layers = [24, 16], dropout_salary = [1.0, 1.0, 1.0], salary_prototype = 21, difficulty_encoder_layers = [24, 16], difficulty_decoder_layers = [24, 16], dropout_difficulty = [1.0, 1.0, 1.0], difficulty_prototype = 21, dropout_prototype = [1.0, 1.0, 1.0], activation = tf.nn.relu, random_seed = 2022,l2_reg = 0.0, name="", pool="sum", sess=None):
        
        self.n_s = n_s
        self.lambda_d = lambda_d
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.embsize = embsize
        self.pool_size = pool_size
        self.lambda_info = lambda_info
        self.lambda_div = lambda_div
        self.lambda_enc = lambda_enc
        self.lambda_freq = lambda_freq

        self.salary_encoder_layers = salary_encoder_layers
        self.salary_decoder_layers = salary_decoder_layers
        self.dropout_salary_feed = dropout_salary
        self.salary_prototype = salary_prototype
        self.difficulty_encoder_layers = difficulty_encoder_layers
        self.difficulty_decoder_layers = difficulty_decoder_layers
        self.dropout_difficulty_feed = dropout_difficulty
        self.difficulty_prototype = difficulty_prototype
        self.dropout_prototype_feed = dropout_prototype

        self.activation = activation
        self.random_seed = random_seed
        self.l2_reg = l2_reg
        self.name = name
        # pooling methods could be "sum", "max", "mean"
        self.pool = pool
        self.sess = sess

        self.weights = {}
        # self.attention = {}
        self.dropouts = {}
        self._init_graph()

    def _init_graph(self):
        tf.set_random_seed(self.random_seed)
        self._build_network()
        #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate).minimize(self.loss, var_list=trainable_lst)
        # init 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.saver = tf.train.Saver(max_to_keep=1)
        init = tf.global_variables_initializer()

        # freeze the following model parameters
        #names = {'salary_embedding', 'salary_skill_action_value', 'salary_prototype', 'difficulty_embedding', 
        #        'difficulty_skill_action_value', 'difficulty_prototype'}

        #trainable_lst = []
        #for item in tf.trainable_variables():
        #    if not any(name in item.name for name in names):
        #        trainable_lst.append(item)
         
        #self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate).minimize(self.loss, var_list=trainable_lst)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        if self.sess is None:
            self.sess = tf.Session(config=config)
            self.sess.run(init)

        print("#params: %d" % self._count_parameters())
            

    def run(self, data, batch_size, train=True):
        n_data = len(data[0])
        predictions_salary, predictions_easy, loss = [], [], []
        info_loss, div_loss, freq_loss, enc_loss = [], [], [], []
        for i in range(0, n_data, batch_size):
            # data_batch = [dt[i:min(i + batch_size, n_data)] for dt in data]
            data_batch = data
            if train:
                preds_easy, preds_salary, loss_step, info_loss_step, div_loss_step, freq_loss_step, enc_loss_step , _ = self.sess.run((self.v_difficulty, self.v_salary, self.loss, self.info_loss, self.diversity_loss, self.frequency_loss, self.encoder_loss, self.optimizer), feed_dict=self.get_dict(data_batch, train=True))
            else:
                # retrieve the corresponding data run( , )
                preds_easy, preds_salary = self.sess.run((self.v_difficulty, self.v_salary),
                                                         feed_dict=self.get_dict(data_batch, train=False))
            predictions_salary.extend(preds_salary)
            predictions_easy.extend(preds_easy)
            loss.append(loss_step)

            info_loss.append(info_loss_step)
            div_loss.append(div_loss_step)
            freq_loss.append(freq_loss_step)
            enc_loss.append(enc_loss_step)

        return predictions_salary, predictions_easy, loss, info_loss, div_loss, freq_loss, enc_loss

    def get_dict(self, data, train=True):
        feed_dict = {
            self.input_state: data[0],
            self.input_action: data[1],
            self.salary_label: data[2],
            self.difficulty_label: data[3],
            self.frequency_set: data[4],
            #self.deep_encoder_salary_action_dropout: self.dropout_salary_feed if train else [1] * len(self.dropout_salary_feed),
            #self.deep_encoder_difficulty_action_dropout: self.dropout_difficulty_feed if train else [1] * len(self.dropout_difficulty_feed),
            self.deep_decoder_salary_dropout: self.dropout_salary_feed if train else [1] * len(self.dropout_salary_feed),
            self.deep_decoder_difficulty_dropout: self.dropout_difficulty_feed if train else [1] * len(self.dropout_difficulty_feed),
            self.is_training: train
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
        self.salary_label = tf.placeholder(tf.float32, shape = [None], name = "%s_input_salary_label" % self.name)
        self.difficulty_label = tf.placeholder(tf.float32, shape = [None], name = "%s_input_difficulty_label" % self.name)
        self.input_action = tf.placeholder(tf.int32, shape=[None, self.pool_size], name="%s_input_action" % self.name) # num, pool_size
        self.frequency_set = tf.placeholder(tf.float32, shape=[None, self.n_s], name = "%s_frequency_set" % self.name) # num * 10, n_s
        self.is_training = tf.placeholder(tf.bool, name="is_training")


        # ----------------------- multi-hot processing ----------------------
        self.batch_size = tf.shape(self.input_state)[0]
        # input_set = tf.reshape(tf.tile(self.input_state, [1, self.pool_size]), [-1,  self.n_s]) # num * pool_size, n_s

        # action_set = tf.reshape(tf.one_hot(tf.reshape(self.input_action, [-1, 1]), self.n_s), [-1, self.n_s]) # batch_size * pool_size, n_s
        action_set = tf.one_hot(self.input_action, self.n_s) # batch_size, pool_size, n_s

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


        # ----------------------------------------------------------------
        # ---------------------- Part of Salary --------------------------
        # ----------------------------------------------------------------
        # ---------------------- encoder network -------------------------
        # self.deep_salary_dropout, y_deep_encoder, _ = self._mlp(mix_input, self.n_s, self.salary_encoder_layers,
        #                                                name="%s_deep_encoder_salary" % self.name,
        #                                                activation=self.activation, bias=True, sparse_input=True, batch_norm=False)
        
        input_state_expanded = tf.expand_dims(self.input_state, -1) # batch_size, n_s, 1
        salary_emb_expanded = tf.expand_dims(self.salary_emb, 0) # 1, n_s, embsize

        # salary_embedding_mix, _ = self._fc(y_deep_encoder, self.salary_encoder_layers[-1], self.embsize, name="%s_salary_encoder_emb" % self.name, l2_reg=0.0, activation=None, bias=True, sparse=False) 
        if self.pool == "sum":
            input_salary_emb = tf.reduce_sum(tf.multiply(input_state_expanded, salary_emb_expanded), axis=1)
        elif self.pool == "max":
            input_salary_emb = tf.reduce_max(tf.multiply(input_state_expanded, salary_emb_expanded), axis=1)
        elif self.pool == "mean":
            input_salary_emb = tf.reduce_mean(tf.multiply(input_state_expanded, salary_emb_expanded), axis=1)

        # batch_size, embsize
        # self.input_emb = tf.broadcast_to(tf.reshape(input_emb, [-1, 1, self.embsize]), [self.batch_size, self.pool_size, self.embsize]) # batch_size, pool_size, embsize
        
        # add transformation layer
        # input_salary_emb = self._mlp(input_salary_emb, self.embsize, [self.embsize, self.embsize], name="%s_salary_input_state_emb" % self.name, activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        # self.input_salary_emb = tf.tile(tf.reshape(input_salary_emb, [-1, 1, self.embsize]), [1, self.pool_size, 1]) # batch_size, pool_size, embsize
        self.input_salary_emb = tf.reshape(input_salary_emb, [-1, 1, self.embsize]) # batch_size, 1, embsize
        self.raw_input_salary_emb = tf.reshape(input_salary_emb, [-1, 1, self.embsize]) # batch_size, 1, embsize

        self.freq_salary_emb  = tf.matmul(self.frequency_set, self.salary_emb)
        self.action_salary_emb = tf.matmul(action_set, self.salary_emb) # batch_size, pool_size, embsize

        # self.salary_embedding = tf.slice(salary_embedding_mix, [0, 0], [len_input_set, -1])
        # self.salary_embedding_freq = tf.slice(salary_embedding_mix, [len_input_set, 0], [-1, -1])

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
        

        # ---------------------------------------------------------------
        # ---------------------- Part of Difficulty ---------------------
        # ---------------------------------------------------------------

        # ----------------------- difficulty skill embeddings -------------------------
        if "%s_difficulty_embedding" % self.name not in self.weights:
            self.dif_emb = tf.Variable(np.random.normal(size=(self.n_s, self.embsize)),
                          dtype=np.float32, name="%s_difficulty_embedding" % self.name)
        self.weights["%s_difficulty_embedding" % self.name] = self.dif_emb

        # ----------------------- difficulty action weights -----------------------
        if "%s_difficulty_skill_action_value" % self.name not in self.weights:
            self.difficulty_skill_action_value = tf.Variable(np.random.normal(size=(self.n_s, self.difficulty_prototype)),
                            dtype=np.float32, name="%s_difficulty_skill_action_value" % self.name)
        self.weights["%s_difficulty_skill_action_value" % self.name] = self.difficulty_skill_action_value


        dif_emb_expanded = tf.expand_dims(self.dif_emb, 0) # 1, n_s, embsize

        if self.pool == "sum":
            input_dif_emb = tf.reduce_sum(tf.multiply(input_state_expanded, dif_emb_expanded), axis=1)
        elif self.pool == "max":
            input_dif_emb = tf.reduce_max(tf.multiply(input_state_expanded, dif_emb_expanded), axis=1)
        elif self.pool == "mean":
            input_dif_emb = tf.reduce_mean(tf.multiply(input_state_expanded, dif_emb_expanded), axis=1)

        # input_dif_emb = tf.matmul(self.input_state, dif_emb)
        # self.input_emb = tf.broadcast_to(tf.reshape(input_emb, [-1, 1, self.embsize]), [self.batch_size, self.pool_size, self.embsize]) # batch_size, pool_size, embsize
        
        # add transformation layer
        # input_dif_emb = self._mlp(input_dif_emb, self.embsize, [self.embsize, self.embsize], name="%s_difficulty_input_state_emb" % self.name, activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        # self.input_dif_emb = tf.tile(tf.reshape(input_dif_emb, [-1, 1, self.embsize]), [1, self.pool_size, 1]) # batch_size, pool_size, embsize
        self.input_dif_emb = tf.reshape(input_dif_emb, [-1, 1, self.embsize]) # batch_size, 1, embsize
        self.raw_input_dif_emb = tf.reshape(input_dif_emb, [-1, 1, self.embsize]) # batch_size, 1, embsize

        self.freq_dif_emb  = tf.matmul(self.frequency_set, self.dif_emb)

        self.action_dif_emb = tf.matmul(action_set, self.dif_emb) # batch_size, pool_size, embsize


        # debug unit
        # check0 = tf.debugging.assert_equal(self.salary_embedding_freq, self.difficulty_embedding_freq)

        # ---------------------- decoder network ------------------
        self.deep_decoder_difficulty_dropout , y_dif_deep_decoder, _ = self._mlp(self.freq_dif_emb, self.embsize, self.difficulty_decoder_layers,
                                                        name="%s_deep_decoder_difficulty" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        self.difficulty_decoder, _ = self._fc(y_dif_deep_decoder, self.difficulty_decoder_layers[-1], self.n_s, name="%s_difficulty_decoder_output" % self.name, l2_reg=0.0, activation= tf.math.sigmoid, bias=True, sparse=False)

        # ---------------------- decoder network loss function ------------------
        # self.encoder_loss += self.lambda_d * bce(self.frequency_set, self.difficulty_decoder)
        # self.encoder_loss += self.lambda_d * tf.norm(self.difficulty_decoder, ord=1)
        self.encoder_loss += self.lambda_d * tf.reduce_mean(tf.keras.losses.binary_crossentropy(self.frequency_set, self.difficulty_decoder))

        if "%s_difficulty_prototype" % self.name not in self.weights:
            self.weights["%s_difficulty_prototype" % self.name] =  tf.Variable(tf.random_uniform(shape=[self.difficulty_prototype, self.embsize], dtype=tf.float32), name = "%s_difficulty_prototype" % self.name) # prototype_num, emdsize

        # ---------------------- action network ------------------ 
        # concat the input state and action
        # difficulty_action_input = tf.concat([tf.tile(tf.reshape(self.input_dif_emb, [-1, 1, self.embsize]), [1, self.pool_size, 1]), self.action_dif_emb], axis=-1) # batch_size, pool_size, 2 * embsize

        #self.deep_encoder_difficulty_action_dropout, y_deep_action, _ = self._mlp(self.action_dif_emb, self.embsize, [self.embsize, self.embsize],
        #                                                name="%s_deep_encoder_difficulty_action" % self.name,
        #                                                activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        #self.difficulty_action_value, _ = self._fc(y_deep_action, self.difficulty_encoder_layers[-1], self.difficulty_prototype, name="%s_difficulty_encoder_action" % self.name, l2_reg=0.0, activation = tf.nn.softplus, bias=True, sparse=False, batch_norm=False) 

        # get the difficulty action value in the batch size and pool size
        # batch_size, pool_size, prototype_num
        # action_set: batch_size, pool_size, n_s
        # difficulty_skill_action_value: n_s, prototype_num
        self.difficulty_skill_action_value_softplus = tf.nn.softplus(self.difficulty_skill_action_value)
        self.difficulty_action_value = tf.matmul(action_set, self.difficulty_skill_action_value_softplus) # batch_size, pool_size, prototype_num
        # self.difficulty_action_value = tf.reshape(self.difficulty_action_value, [-1, self.difficulty_prototype])
        

        # add relu constraints
        # self.difficulty_action_value = tf.keras.activations.relu(self.difficulty_action_value, max_value=self.difficulty_q_value)

        # debug
        # self.difficulty_action_value = tf.multiply(self.difficulty_action_value, tf.ones([6400, self.difficulty_prototype], tf.float32))
        # assert_op0 = tf.assert_equal(self.difficulty_prototype, 20)
        # assert_op1 = tf.debugging.assert_shapes([6400, self.difficulty_prototype], data=self.difficulty_action_value)
        # assert_op2 = tf.assert_equal(self.embsize, 256)

        # debug
        # test = tf.ones([6400, 1, 1], tf.float32)
        # opx = tf.assert_equal(y_deep_action.shape, test.shape)

        # ---------------------- difficulty prototype ---------------------
        # apply the same transformation layer on the each part
        # self.difficulty_input_state_emb_dropout, self.input_dif_emb, _ = self._mlp(self.input_dif_emb, self.embsize, [self.embsize], name="%s_difficulty_input_state_emb" % self.name, activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        # self.input_dif_emb, _ = self._fc(self.input_dif_emb, self.embsize, self.embsize, name="%s_difficulty_fc_input_state_emb" % self.name, l2_reg=0.0, activation=None, bias=True, sparse=False)

        self.q_difficulty, difficulty_diversity_loss, difficulty_frequenct_loss, _, difficulty_weights, self.skill_difficulty_attention = self._prototype_layer(self.input_dif_emb, self.freq_dif_emb, self.difficulty_prototype, self.embsize, self.difficulty_action_value, salary_flag=False, name="%s_difficulty" % self.name)
        
        difficulty_infoNCE_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.difficulty_action_value), axis=-1))

        self.q_difficulty = tf.negative(tf.reshape(self.q_difficulty, [-1, self.pool_size]))
        self.frequency_loss += self.lambda_d * difficulty_frequenct_loss
        self.info_loss += self.lambda_d * difficulty_infoNCE_loss
        
        
        # ---------------------- mapping prototypes back ------------------
        _ , y_deep_prototype, _ = self._mlp(difficulty_weights, self.embsize, self.difficulty_decoder_layers,
                                                        name="%s_deep_decoder_difficulty" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        self.skill_difficulty_prototypes, _ = self._fc(y_deep_prototype, self.difficulty_decoder_layers[-1], self.n_s, name="%s_difficulty_decoder_output" % self.name, l2_reg=0.0, activation= tf.nn.sigmoid, bias=True, sparse=False) # prototype_num * n_s
        

        # ---------------------------------------------------------------------
        # --------------------- Choose an action ------------------------------
        # ---------------------------------------------------------------------
        self.q = self.q_salary + self.lambda_d * self.q_difficulty # -1, pool_size    

        self.v_place = tf.argmax(self.q, 1)

        onehot_action_nxt = tf.one_hot(self.v_place, self.pool_size)  # should be: data batch, pool_size  

        self.v_skill = tf.cast(tf.reduce_sum(tf.multiply(onehot_action_nxt, tf.cast(self.input_action, np.float32)), 1), np.int32)        
        self.v = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q), 1)
        self.v_salary = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q_salary), 1)
        self.v_difficulty = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q_difficulty), 1)

        
        # ------ attention selections-----------------------
        self.skill_salary_attention = tf.tile(tf.reshape(self.skill_salary_attention, [-1, 1, self.salary_prototype]), [1, self.pool_size, 1]) # batch_size, pool_size, prototype_num
        self.skill_difficulty_attention = tf.tile(tf.reshape(self.skill_difficulty_attention, [-1, 1, self.difficulty_prototype]), [1, self.pool_size, 1]) # batch_size, pool_size, prototype_num   

        # att_onehot_salary = tf.broadcast_to(tf.reshape(onehot_action_nxt, [-1, self.pool_size, 1]), [self.batch_size, self.pool_size, self.salary_prototype])# should be: data batch, pool_size, prototype_num
        # att_onehot_difficulty = tf.broadcast_to(tf.reshape(onehot_action_nxt, [-1, self.pool_size, 1]), [self.batch_size, self.pool_size, self.difficulty_prototype])# should be: data batch, pool_size, prototype_num

        #att_onehot_salary = tf.tile(tf.reshape(onehot_action_nxt, [-1, self.pool_size, 1]), [1, 1, self.salary_prototype])# should be: data batch, pool_size, prototype_num
        #att_onehot_difficulty = tf.tile(tf.reshape(onehot_action_nxt, [-1, self.pool_size, 1]), [1, 1, self.difficulty_prototype])# should be: data batch, pool_size, prototype_num
        att_onehot_salary = tf.reshape(onehot_action_nxt, [-1, self.pool_size, 1])# should be: data batch, pool_size, 1
        att_onehot_difficulty = tf.reshape(onehot_action_nxt, [-1, self.pool_size, 1])# should be: data batch, pool_size, 1

        self.skill_salary_attention_final = tf.reduce_sum(tf.multiply(att_onehot_salary, self.skill_salary_attention), axis=1)
        self.skill_difficulty_attention_final = tf.reduce_sum(tf.multiply(att_onehot_difficulty, self.skill_difficulty_attention), axis=1)

        # --------------------- loss ----------------------------------
        # --------------------- need to add lambda_d to restrict ------
        self.diversity_loss = tf.reduce_mean(salary_diversity_loss)
        self.diversity_loss += self.lambda_d * tf.reduce_mean(difficulty_diversity_loss)

        self.loss = tf.reduce_mean(tf.square(self.v_salary - self.salary_label))
        self.loss += self.lambda_d * tf.reduce_mean(tf.square(self.v_difficulty - self.difficulty_label))
        self.loss += self.lambda_div * self.diversity_loss
        self.loss += self.lambda_enc * self.encoder_loss
        self.loss += self.lambda_freq * self.frequency_loss
        self.loss += self.lambda_info * self.info_loss
        return 

    #def _similarity_function(self, tensor1, tensor2):
        # euclidean distance
    #    distance = tf.norm(tf.cast(tensor1 - tensor2, tf.float32), ord='euclidean', axis=-1)
    #    return tf.math.log((distance + 1)/(distance + tf.keras.backend.epsilon()))

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

    def _diversity_loss(self, prototypes, d_min):
        # Expand the prototype tensors to calculate pairwise differences
        p1 = tf.expand_dims(prototypes, 1)  # Shape: [prototype_num, 1, embedding_size]
        p2 = tf.expand_dims(prototypes, 0)  # Shape: [1, prototype_num, embedding_size]

        # Calculate pairwise squared L2 distances
        pairwise_distances_squared = tf.reduce_sum(tf.square(p1 - p2), axis=-1)  # Shape: [prototype_num, prototype_num]

        # Apply the loss formula
        loss = tf.maximum(0.0, d_min - pairwise_distances_squared)

        # Zero out the diagonal (distance of each prototype to itself), as it's not needed
        mask = 1 - tf.eye(tf.shape(prototypes)[0])
        loss *= mask

        # Aggregate the loss (e.g., mean or sum)
        total_loss = tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1)
    
        return total_loss
    
    # contain the prototype duplicates
    def _prototype_layer(self, tensor, frequency_set, prototype_size, embedding_size, q_value, salary_flag, name):
        ### q_value: batch_size, pool_size, prototype_num
        ### tensor(input state): batch_size, 1, embedding_size
        ### frequency_set: -1, embedding_size

        #if "%s_prototype" % name not in self.weights:
        #    self.weights["%s_prototype" % name] =  tf.Variable(tf.random_uniform(shape=[prototype_size, embedding_size], dtype=tf.float32), name = #"%s_prototype" % name) # prototype_num, emdsize
        
        # apply the same transformation layer on the each part
        prototypes = self.weights["%s_prototype" % name]

        # prototypes, _ = self._fc(deep_prototypes, embedding_size, embedding_size, name="%s_difficulty_encoder_action" % name, l2_reg=0.0, activation = tf.nn.relu, bias=True, sparse=False)

        # tensor_prot = tf.broadcast_to(tf.reshape(tensor, [-1, 1, embedding_size]), [self.batch_size * self.pool_size, prototype_size, self.embsize]) # num * pool_size, prototype_num, emdsize

        # tensor_prot = tf.tile(tf.reshape(tensor, [-1, 1, embedding_size]), [1, prototype_size, 1])
        tensor_prot = tf.expand_dims(tensor, 2) # batch_size, 1, 1, embedding_size
        prototypes = tf.expand_dims(prototypes, 0) # 1, prototype_num, embedding_size
        # print(tensor_prot.get_shape(), prototypes.get_shape())

        distance = self._l2_distance_function(tensor_prot, tf.expand_dims(prototypes, 0)) # batch_size, 1, prototype_num
        neg_distance = tf.exp(-self._l2_distance_function(tensor_prot, tf.expand_dims(prototypes, 0))) # batch_size, 1, prototype_num
        # cancel softmax
        # similarity = tf.nn.softmax(similarity, axis=-1) # batch_size, 1, prototype_num

        #self.attention["%s_prototype_weights" % name] = tf.nn.softmax(similarity, axis=1) # -1, prototype_num
        # attention = tf.nn.softmax(similarity, axis=1) # -1, prototype_num
        
        # debug unit
        # check0 = tf.debugging.assert_equal(similarity, q_value)
        q_result = tf.reduce_sum(tf.multiply(neg_distance, q_value), axis=-1, keepdims=True) # batch_size, pool_size, prototype_num
        
        # clusering and evidence loss
        frequency_loss = 0
        
        # freq_prot = tf.broadcast_to(tf.reshape(frequency_set, [-1, 1, embedding_size]), [self.batch_size * 4, prototype_size, self.embsize]) # num * sampling size, prototype_num, emdsize
        

        # for each frequency set 
        freq_min_loss = tf.reduce_mean(tf.reduce_min(distance, axis=1))
        frequency_loss += freq_min_loss

        # for each prototype
        evidence_prototype_loss = tf.reduce_mean(tf.reduce_min(distance, axis=0))
        frequency_loss += evidence_prototype_loss
        
        # infoNCE loss 
        # for each frequency set
        # infoNCE_loss = tf.math.log(tf.reduce_min(freq_distance, axis=1)/tf.reduce_sum(freq_distance, axis=1))
        infoNCE_loss = tf.convert_to_tensor(0, dtype=tf.float32)

        # diversity loss
        # max(0, d_min - ||p_i - p_j||_2^2) over prototypes
        diversity_loss = self._diversity_loss(tf.squeeze(prototypes), 1.0)

        return q_result, diversity_loss, frequency_loss, infoNCE_loss, self.weights["%s_prototype" % name], neg_distance


    def _fc(self, tensor, dim_in, dim_out, name, l2_reg, activation=None, bias=True, sparse=False, batch_norm=False):
        glorot = np.sqrt(2.0 / (dim_in + dim_out))
        # Weight initialization
        if "%s_w" % name not in self.weights:
            self.weights["%s_w" % name] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(dim_in, dim_out)),
                                                      dtype=np.float32, name="%s_w" % name)
        # Matrix multiplication
        if not sparse:
            y_deep = tf.matmul(tensor, self.weights["%s_w" % name])
        else:
            y_deep = tf.sparse_tensor_dense_matmul(tensor, self.weights["%s_w" % name])
        # Bias addition
        if bias:
            if "%s_b" % name not in self.weights:
                self.weights["%s_b" % name] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, dim_out)),
                                                          dtype=np.float32, name="%s_b" % name)
                y_deep += self.weights["%s_b" % name]

        # Batch Normalization
        if batch_norm:
            y_deep = tf.layers.batch_normalization(y_deep, training=self.is_training)

        # Activation function
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
                                            bias=bias, activation=activation, sparse=True, batch_norm=batch_norm)
            else:
                y_deep, _ = self._fc(y_deep, dim_in, layer, l2_reg=self.l2_reg, name="%s_%d" % (name, i),
                                            bias=bias, activation=activation, sparse=False, batch_norm=batch_norm)
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
        data = [[state_vis], [act_pool], [0], [0], [[0] * self.n_s]]
        act, v_salary, v_easy = self.sess.run((self.v_skill, self.v_salary, self.v_difficulty), self.get_dict(data, train=False))
        return (v_salary[0], v_easy[0]), act[0]

    def estimate_maxq_batch(self, data_state, data_pool):
        n_data = len(data_state)
        act_lst, v_salary, v_easy = self.sess.run((self.v_skill, self.v_salary, self.v_difficulty),
                                                  self.get_dict((data_state, data_pool, [0] * n_data, [0] * n_data, [[0] * self.n_s] * n_data * 4), train=False))
        return (v_salary, v_easy), act_lst    

    def estimate_maxq_batch_sample(self, data_state, data_pool):
        # n_data = len(data_state)
        act_lst, q_salary, q_easy, q = self.sess.run((self.input_action, self.q_salary, self.q_difficulty, self.q),
                                                  self.get_dict(([data_state], [data_pool], [0], [0], [[0] * self.n_s]), train=False))
        return (q_salary, q_easy), act_lst, q
    
    def estimate_maxq_action_with_prototypes(self, state_vis, act_pool):
        # here, we radomly select the frequent set to test the decoder accuracy
        data = [[state_vis], [act_pool], [0], [0], [sparse_to_dense([107, 356, 639, 953], self.n_s)]]
        act, v_salary, v_easy, attention_salary, salary_prototypes, attention_difficulty, difficulty_prototypes, salary_action_value, difficulty_action_value, salary_mapped, original_salary_mapped, v_place, difficulty_mapped = self.sess.run((
                                               self.v_skill, self.v_salary, self.v_difficulty, 
                                               self.skill_salary_attention_final, self.skill_salary_prototypes,
                                               self.skill_difficulty_attention_final, self.skill_difficulty_prototypes, self.salary_action_value, self.difficulty_action_value, self.salary_decoder, self.frequency_set, self.v_place, self.difficulty_decoder), self.get_dict(data, train=False))

        return (v_salary[0], v_easy[0]), act[0], attention_salary, salary_prototypes, attention_difficulty, difficulty_prototypes, salary_action_value, difficulty_action_value, salary_mapped, original_salary_mapped, v_place, difficulty_mapped

    def estimate_maxq_action_with_prototypes_with_skill(self, state_vis, s):
        # here, we radomly select the frequent set to test the decoder accuracy
        data = [[state_vis], [[s] * self.pool_size], [0], [0], [sparse_to_dense([107, 356, 639, 953], self.n_s)]]
        act, v_salary, v_easy, attention_salary, salary_prototypes, attention_difficulty, difficulty_prototypes, salary_action_value, difficulty_action_value, salary_mapped, original_salary_mapped, v_place, difficulty_mapped = self.sess.run((
                                               self.v_skill, self.v_salary, self.v_difficulty, 
                                               self.skill_salary_attention_final, self.skill_salary_prototypes,
                                               self.skill_difficulty_attention_final, self.skill_difficulty_prototypes, self.salary_action_value, self.difficulty_action_value, self.salary_decoder, self.frequency_set, self.v_place, self.difficulty_decoder), self.get_dict(data, train=False))

        return (v_salary[0], v_easy[0]), act[0], attention_salary, salary_prototypes, attention_difficulty, difficulty_prototypes, salary_action_value, difficulty_action_value, salary_mapped, original_salary_mapped, v_place, difficulty_mapped

    def evaluate_internal_value_functions(self, state_vis, act_pool):
        data = [[state_vis], [act_pool], [0], [0], [sparse_to_dense([416, 222, 183, 743, 746], self.n_s)]]
        loss, encoder_loss, div_loss, freq_loss = self.sess.run((self.loss, self.encoder_loss, self.diversity_loss, self.frequency_loss), self.get_dict(data, train=False))
        return loss, encoder_loss, div_loss, freq_loss
    
    def get_q_list(self, skill_vis, act_pool):
        data = [[skill_vis], [act_pool], [0], [0], [[0] * self.n_s]]
        q_salary_lst, q_easy_lst = self.sess.run((self.q_salary, self.q_difficulty), self.get_dict(data, train=False))
        return q_salary_lst[0], q_easy_lst[0]

    #TODD: remove the variable flag
    def get_embeddings(self, state_vis, flag):
        input_state = [sparse_to_dense(state_vis, self.n_s)]
        data = [input_state, [[0] * self.pool_size], [0], [0], [[0] * self.n_s]]
        if flag: 
           embeddings = self.sess.run((self.input_salary_emb), self.get_dict(data, train=False))
        else:
           embeddings = self.sess.run((self.input_dif_emb), self.get_dict(data, train=False))
        return embeddings[0][0]
    
    def get_raw_embeddings(self, state_vis, flag):
        input_state = [sparse_to_dense(state_vis, self.n_s)]
        data = [input_state, [[0] * self.pool_size], [0], [0], [[0] * self.n_s]]
        if flag: 
           embeddings = self.sess.run((self.raw_input_salary_emb), self.get_dict(data, train=False))
        else:
           embeddings = self.sess.run((self.raw_input_dif_emb), self.get_dict(data, train=False))
        return embeddings[0][0]

    def get_salary_prototypes(self):
        data = [[[0] * self.n_s], [[0] * self.pool_size], [0], [0], [[0] * self.n_s]]
        embeddings, decoder_result = self.sess.run((self.weights["%s_salary_prototype" % self.name], self.skill_salary_prototypes), self.get_dict(data, train=False))
        # decoder_result = self.sess.run((self.salary_decoder), self.get_dict(data, train=False))
        return decoder_result, embeddings

    def get_difficulty_prototypes(self):
        data = [[[0] * self.n_s], [[0] * self.pool_size], [0], [0], [[0] * self.n_s]]
        embeddings, decoder_result = self.sess.run((self.weights["%s_difficulty_prototype" % self.name], self.skill_difficulty_prototypes), self.get_dict(data, train=False))
        return decoder_result, embeddings
