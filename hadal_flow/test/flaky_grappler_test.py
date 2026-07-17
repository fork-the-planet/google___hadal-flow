import tempfile
import unittest
import numpy as np
import tensorflow as tf
import hadal_flow as hadal
import hadal_ml
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate


class FlakyGrapplerTest(tf.test.TestCase):
    def test_flaky_grappler(self):
        hadal.enable_optimization()
        failed = False

        context = hadal.create_context64(
            log_n=11,
            main_moduli=[8556589057, 8388812801],
            plaintext_modulus=40961,
            scaling_factor=1,
        )

        weights = [tf.random.uniform((10, 10)) for _ in range(30)]
        for i in range(10):

            @tf.function
            def predict_fn(features, context):
                shell_tensors = []
                # First loop: all encodes
                for j in range(30):
                    a = tf.matmul(features, weights[j])
                    a_shell = hadal.to_shell_plaintext(a, context)
                    shell_tensors.append(a_shell)

                res = []
                # Second loop: all decodes
                # Use control dependencies to force ALL encodes to complete before ANY decode starts.
                # This explicitly groups all 'Encode' nodes before 'Decode' nodes topologically.
                with tf.control_dependencies(shell_tensors):
                    for j in range(30):
                        a_tf = hadal.to_tensorflow(shell_tensors[j])
                        res.append(a_tf)

                return tf.add_n(res)

            func = predict_fn.get_concrete_function(
                tf.TensorSpec((4, 10), dtype=tf.float32), context._type_spec
            )

            optimized_func = hadal.optimize_shell_graph(func)

            graph_def = optimized_func.graph.as_graph_def()
            ops_present = set(node.op for node in graph_def.node)
            print("OPS IN OPTIMIZED GRAPH:")
            print(ops_present)

            # check original graph ops
            orig_graph_def = func.graph.as_graph_def()
            print("OPS IN ORIG GRAPH:")
            print(set(node.op for node in orig_graph_def.node))

            decode_count = sum(
                1
                for node in graph_def.node
                if node.op == "PolynomialExport64" or node.op == "PolynomialImport64"
            )

            if decode_count > 0:
                print(f"FAILED on iteration {i} with {decode_count} leftover ops!")
                sys.stdout.flush()
                failed = True
                break

        self.assertFalse(failed, "Flaky grappler optimization bug triggered!")


if __name__ == "__main__":
    tf.test.main()
