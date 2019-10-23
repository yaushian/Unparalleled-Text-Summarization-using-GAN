"""
This is the module that given long text sequence generates short text sequence
"""
import tensorflow as tf
from lib.ops import *


def generator(
            encoder_inputs,
            vocab_size,
            word_embedding_matrix,
            encoder_length,
            decoder_inputs,
            feed_previous,
            do_sample=False,
            do_beam_search=False,
            latent_dim=300):
    """
    a sequence to sequence pointer network model 
    inputs should be word id and outputs will be softmax over words
    """
    
    #initialize for beam search
    if do_beam_search:
        beam_size = 5
        batch_size = encoder_inputs.get_shape().as_list()[0]
        decoder_preds = []
        #the log probability for each path
        path_probs = tf.concat([tf.zeros([batch_size,1]),tf.zeros([batch_size,beam_size-1])-10.],axis=1)
        batch_start_id = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size),[-1,1]),[1,beam_size]),[-1])*beam_size
        encoder_inputs = tf.reshape(tf.tile(encoder_inputs,[1,beam_size]),[batch_size*beam_size,-1])
        encoder_length = encoder_length*beam_size
        
        #penalty for long or short sequence
        penalty1 = np.ones([batch_size,beam_size,vocab_size])
        penalty1[:,:,0] = 0.1
        penalty2 = np.ones([batch_size,beam_size,vocab_size])
        penalty2[:,:,0] = 0.5
        penalty3 = np.ones([batch_size,beam_size,vocab_size])

    input_one_hot = tf.one_hot(encoder_inputs,vocab_size)
    input_embedded = tf.nn.embedding_lookup(word_embedding_matrix,encoder_inputs)
    encoder_inputs_embedded = tf.nn.embedding_lookup(word_embedding_matrix,encoder_inputs)
    encoder_shape = encoder_inputs.get_shape().as_list()

    decoder_inputs = batch_to_time_major(decoder_inputs)
    if do_beam_search:
        bos_input = tf.reshape(tf.tile(decoder_inputs[0],[1,beam_size]),[batch_size*beam_size,-1])

    with tf.variable_scope("generator_encoder") as scope:
        cell = tf.contrib.rnn.LSTMCell(num_units=latent_dim*2, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        encoder_outputs,encoder_state = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=encoder_length,
            inputs=encoder_inputs_embedded)

    #pointer network
    with tf.variable_scope("generator_pointer_decoder") as scope:

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
            hidden = tf.layers.dense(tf.concat([state,attention_context],axis=1),1000,name='P_vocab_projection1')
            vocab_weight = tf.layers.dense(hidden,vocab_size,name='P_vocab_projection2')
            return tf.nn.softmax(vocab_weight)


        #outputs is a softmax probability distribution
        decoder_outputs = []
        real_outputs_ids = [] if feed_previous else None
        real_outputs_probs = []
        coverage_loss = []

        #initialize decoder state with encoder state and initialize attention
        state = encoder_state
        attention_coverage = tf.zeros([encoder_shape[0],encoder_shape[1]])
        attention_weight,attention_context = do_attention(state.h,attention_coverage)

        for i in range(len(decoder_inputs)):
            if i > 0:
                scope.reuse_variables()
            if i==0 and do_beam_search:
                input_t = bos_input
            elif i>0 and do_beam_search:
                if i<=7:
                    penalty = np.log(penalty1)
                elif i<=10:
                    penalty = np.log(penalty2)
                else:
                    penalty = np.log(penalty3)
                out_probs = tf.reshape(decoder_outputs[-1], [batch_size,beam_size,vocab_size])
                new_path_prob = tf.reshape(path_probs,[batch_size,beam_size,1]) + tf.log(out_probs+1e-7) + penalty
                new_path_prob = tf.reshape(new_path_prob,[batch_size, beam_size*vocab_size])
                path_probs,top_ids = tf.nn.top_k(new_path_prob, beam_size)
                real_outputs_ids.append(top_ids)
                real_outputs_probs.append(path_probs)

                #change the state corresponding to correct state
                top_batch_ids = batch_start_id + tf.floordiv(tf.reshape(top_ids,[batch_size*beam_size]),vocab_size)
                new_s_c,new_s_h = tf.gather(state.c,top_batch_ids),tf.gather(state.h,top_batch_ids)
                state = tf.nn.rnn_cell.LSTMStateTuple(new_s_c,new_s_h)
                attention_coverage = tf.gather(attention_coverage,top_batch_ids)
                attention_context = tf.gather(attention_context,top_batch_ids)
                
                #output for next time step
                last_output_id = tf.reshape(tf.mod(top_ids,int(vocab_size)),[batch_size*beam_size])
                input_t = tf.nn.embedding_lookup(word_embedding_matrix,last_output_id)
            elif i > 0 and feed_previous and do_sample:
                last_output_id = sample2D(decoder_outputs[-1])
                batch_id = tf.cast(tf.range(encoder_shape[0]),dtype=tf.int64)
                abs_id = tf.concat([tf.expand_dims(batch_id,axis=1),tf.expand_dims(last_output_id,axis=1)],axis=1)
                last_output_prob = tf.gather_nd(decoder_outputs[-1],abs_id)
                input_t = tf.nn.embedding_lookup(word_embedding_matrix,last_output_id)
            elif i > 0 and feed_previous:
                last_output_id = tf.argmax(decoder_outputs[-1],axis=-1)
                batch_id = tf.range(encoder_shape[0])
                abs_id = tf.concat([tf.expand_dims(last_output_id,axis=1),tf.expand_dims(last_output_id,axis=1)],axis=1)
                last_output_prob = tf.gather_nd(decoder_outputs[-1],abs_id)
                input_t = tf.nn.embedding_lookup(word_embedding_matrix,last_output_id)
            else:
                input_t = decoder_inputs[i]

            if feed_previous and i>0 and not do_beam_search:
                real_outputs_ids.append(last_output_id)
                real_outputs_probs.append(last_output_prob)

            cell_ouput,state = cell(input_projection(input_t,attention_context),state)
            attention_weight,attention_context = do_attention(state.h,attention_coverage)
            coverage_loss.append(tf.minimum(attention_coverage,attention_weight))
            attention_coverage += attention_weight

            P_gen = tf.sigmoid(tf.layers.dense( tf.concat([input_t,state.h,attention_context], axis=1), 1, name='P_gen'))
            output_t = P_gen*get_vocab_distribution(state.h,attention_context) + (1 - P_gen)*get_pointer_distribution(attention_weight)
            #decoder outputs: shape=[time_step,batch_size,vocab_size] value is the probability of vocabulary         
            decoder_outputs.append(output_t)

        if do_beam_search:
            out_probs = tf.reshape(decoder_outputs[-1], [batch_size,beam_size,vocab_size])
            new_path_prob = tf.reshape(path_probs,[batch_size,beam_size,1]) + tf.log(out_probs+1e-7)
            new_path_prob = tf.reshape(new_path_prob,[batch_size, beam_size*vocab_size])
            path_probs,top_ids = tf.nn.top_k(new_path_prob, beam_size)
            real_outputs_ids.append(top_ids)
            real_outputs_probs.append(path_probs)
        elif feed_previous:
            real_outputs_ids.append(tf.argmax(decoder_outputs[-1],axis=-1))
            real_outputs_probs.append(tf.reduce_max(decoder_outputs[-1],axis=1))
        else:
            real_outputs_ids = [tf.argmax(output,axis=-1) for output in decoder_outputs]
            real_outputs_probs = [tf.reduce_max(output,axis=-1) for output in decoder_outputs]

        print('real outputs ids:',real_outputs_ids[0].get_shape().as_list())
        print('decoder:',len(decoder_outputs),decoder_outputs[0].get_shape().as_list())

    return decoder_outputs,real_outputs_ids,real_outputs_probs,coverage_loss