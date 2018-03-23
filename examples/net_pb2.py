# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: net.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='net.proto',
  package='yantk',
  syntax='proto2',
  serialized_pb=_b('\n\tnet.proto\x12\x05yantk\"\x96\x02\n\x07NetDesc\x12\x19\n\x0enum_iterations\x18\x01 \x01(\x05:\x01\x31\x12>\n\x0cnonlinearity\x18\x02 \x01(\x0e\x32\x1f.yantk.NetDesc.NonlinearityType:\x07SIGMOID\x12\x1a\n\rlearning_rate\x18\x03 \x01(\x02:\x03\x30.1\x12\x14\n\x05\x62\x61tch\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x1f\n\x10report_frequency\x18\x05 \x01(\x05:\x05\x31\x30\x30\x30\x30\x12(\n\x13output_weights_file\x18\x06 \x01(\t:\x0bweights.out\"3\n\x10NonlinearityType\x12\x0b\n\x07SIGMOID\x10\x00\x12\x08\n\x04TANH\x10\x01\x12\x08\n\x04RELU\x10\x02')
)



_NETDESC_NONLINEARITYTYPE = _descriptor.EnumDescriptor(
  name='NonlinearityType',
  full_name='yantk.NetDesc.NonlinearityType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SIGMOID', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TANH', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RELU', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=248,
  serialized_end=299,
)
_sym_db.RegisterEnumDescriptor(_NETDESC_NONLINEARITYTYPE)


_NETDESC = _descriptor.Descriptor(
  name='NetDesc',
  full_name='yantk.NetDesc',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_iterations', full_name='yantk.NetDesc.num_iterations', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='nonlinearity', full_name='yantk.NetDesc.nonlinearity', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='yantk.NetDesc.learning_rate', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='batch', full_name='yantk.NetDesc.batch', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='report_frequency', full_name='yantk.NetDesc.report_frequency', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=10000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output_weights_file', full_name='yantk.NetDesc.output_weights_file', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("weights.out").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _NETDESC_NONLINEARITYTYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21,
  serialized_end=299,
)

_NETDESC.fields_by_name['nonlinearity'].enum_type = _NETDESC_NONLINEARITYTYPE
_NETDESC_NONLINEARITYTYPE.containing_type = _NETDESC
DESCRIPTOR.message_types_by_name['NetDesc'] = _NETDESC
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NetDesc = _reflection.GeneratedProtocolMessageType('NetDesc', (_message.Message,), dict(
  DESCRIPTOR = _NETDESC,
  __module__ = 'net_pb2'
  # @@protoc_insertion_point(class_scope:yantk.NetDesc)
  ))
_sym_db.RegisterMessage(NetDesc)


# @@protoc_insertion_point(module_scope)
