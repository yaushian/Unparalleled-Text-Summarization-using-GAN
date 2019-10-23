"""
This is the module that given long text sequence generates short text sequence
"""
import tensorflow as tf
from lib.ops import *


def reconstructor(
            encoder_inputs,
            vocab_size,
            encoder_length,
            decoder_inputs,
            decoder_targets,
            latent_dim=200):
    """
    a sequence to sequence pointer network model 
    inputs should be word id and outputs will be softmax over words
    """
    init = tf.contrib.layers.xavier_initializer()

    word_embedding_matrix = tf.get_variable(
        name="rec_word_embedding_matrix",
        shape=[vocab_size,100],
        initializer=init,   
        trainable = True
    )

    decoder_inputs = tf.nn.embedding_lookup(word_embedding_matrix,decoder_inputs)
    input_one_hot = tf.one_hot(encoder_inputs,vocab_size)
    input_embedded = tf.nn.embedding_lookup(word_embedding_matrix,encoder_inputs)

    encoder_inputs_embedded = tf.nn.embedding_lookup(word_embedding_matrix,encoder_inputs)
    encoder_shape = encoder_inputs.get_shape().as_list()

    decoder_inputs = batch_to_time_major(decoder_inputs)

    with tf.variable_scope("reconstructor_encoder") as scope:
        fw_cell = tf.contrib.rnn.LSTMCell(num_units=latent_dim, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.LSTMCell(num_units=latent_dim, state_is_tuple=True)
        #bi-lstm encoder
        encoder_outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = fw_cell,
            cell_bw = bw_cell,
            dtype = tf.float32,
            sequence_length = encoder_length,
            inputs = encoder_inputs_embedded,
            time_major=False
        )

        output_fw, output_bw = encoder_outputs
        state_fw, state_bw = state
        encoder_outputs = tf.concat([output_fw,output_bw],2)      #not pretty sure whether to reverse output_bw
        encoder_state_c = tf.concat((state_fw.c, state_bw.c), 1)
        encoder_state_h = tf.concat((state_fw.h, state_bw.h), 1)
        encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

    #pointer network
    with tf.variable_scope("reconstructor_pointer_decoder") as scope:

        #variables
        V = tf.get_variable(name="V", shape=[latent_dim, 1])
        W_h = tf.get_variable(name="W_h", shape=[latent_dim * 2, latent_dim])
        W_s = tf.get_variable(name="W_s", shape=[latent_dim * 2, latent_dim])
        b_attn = tf.get_variable(name="b_attn", shape=[latent_dim])
        w_c = tf.get_variable(name="w_c", shape=[latent_dim])

        #cell
        cell = tf.contrib.rnn.LSTMCell(num_units=latent_dim*2, state_is_tuple=True)

        #functions
        def input_projection(raw_input, last_attention_context):
            return tf.layers.dense(tf.concat([raw_input, last_attention_context], axis=1), latent_dim*2, name="input_projection")


        def do_attention(state,c_t):
            e_t = []
            attention_state = encoder_outputs
            c_t = tf.split(c_t,num_or_size_splits=encoder_shape[1],axis=1)

            for h_i,c_i in zip(batch_to_time_major(attention_state),c_t):
                hidden = tf.tanh(tf.matmul(h_i,W_h) + tf.matmul(state,W_s) + w_c*c_i + b_attn)
                e_t_i = tf.squeeze(tf.matmul(hidden,V),1)
                e_t.append(e_t_i)
            #attention weight shape: batch_size * input_time_step 
            #attention state: batch_size * input_time_step * hidden_size
            attention_weight = tf.nn.softmax(tf.stack(e_t,axis=1))
            attention_context = tf.squeeze(tf.matmul(tf.expand_dims(attention_weight,axis=1),attention_state),axis=1)
            return attention_weight,attention_context


        def get_pointer_distribution(attention_weight):
            return tf.squeeze(tf.matmul(tf.expand_dims(attention_weight,axis=1),input_one_hot),axis=1)


        def get_vocab_distribution(state,attention_context):
            hidden = tf.layers.dense(tf.concat([state,attention_context],axis=1),500,name='P_vocab_projection1')
            vocab_weight = tf.layers.dense(hidden,vocab_size,name='P_vocab_projection2')
            return tf.nn.softmax(vocab_weight)


        #outputs is a softmax probability distribution
        decoder_outputs = []
        #initialize decoder state with encoder state and initialize attention
        state = encoder_state
        attention_coverage = tf.zeros([encoder_shape[0],encoder_shape[1]])
        attention_weight,attention_context = do_attention(state.h,attention_coverage)


        for i in range(len(decoder_inputs)):
            if i > 0:
                scope.reuse_variables()
            input_t = decoder_inputs[i]

            cell_ouput,state = cell(input_projection(input_t,attention_context),state)
            attention_weight,attention_context = do_attention(state.h,attention_coverage)
            attention_coverage += attention_weight

            P_gen = tf.sigmoid(tf.layers.dense( tf.concat([input_t,state.h,attention_context], axis=1), 1, name='P_gen'))
            output_t = P_gen*get_vocab_distribution(state.h,attention_context) + (1 - P_gen)*get_pointer_distribution(attention_weight)         
            decoder_outputs.append(output_t)

    attention_coverage = tf.split(attention_coverage,num_or_size_splits=encoder_shape[1],axis=1)

    with tf.variable_scope("reconstructor_loss") as scope:
        targets = batch_to_time_major(decoder_targets)
        #words_weight = batch_to_time_major(reconstructor_words_weight) 
        total_loss = []
        length = tf.cast(len(targets),dtype=tf.float32)
        for prob_t,target in zip(decoder_outputs,targets):
            target_prob = tf.reduce_max(tf.one_hot(target,vocab_size)*prob_t,axis=-1)
            cross_entropy = -tf.log(tf.clip_by_value(target_prob,1e-10,1.0))
            total_loss.append(cross_entropy)
        total_loss = tf.reshape(tf.reduce_mean(tf.stack(total_loss,axis=1),axis=1),[-1])

    return total_loss,decoder_outputs,attention_coverage