testdir="test_py"
testme="python ../../examples/xor_tf.py"
mkdir -p $testdir
pushd $testdir
for i in $( ls ../test_configs/*.netproto ); do
    $testme $i
done
popd


