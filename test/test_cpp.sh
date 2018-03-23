testdir="test_cpp"
testme="../../build/cygwin-gcc/testnet.exe"
mkdir -p $testdir
pushd $testdir
for i in $( ls ../test_configs/*.netproto ); do
    $testme $i
done
chmod +rwx *.weights
popd


