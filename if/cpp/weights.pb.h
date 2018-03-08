// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: weights.proto

#ifndef PROTOBUF_weights_2eproto__INCLUDED
#define PROTOBUF_weights_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3004000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3004000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
namespace yantk {
class IterationWeights;
class IterationWeightsDefaultTypeInternal;
extern IterationWeightsDefaultTypeInternal _IterationWeights_default_instance_;
class TrainingWeights;
class TrainingWeightsDefaultTypeInternal;
extern TrainingWeightsDefaultTypeInternal _TrainingWeights_default_instance_;
class Weights;
class WeightsDefaultTypeInternal;
extern WeightsDefaultTypeInternal _Weights_default_instance_;
}  // namespace yantk

namespace yantk {

namespace protobuf_weights_2eproto {
// Internal implementation detail -- do not call these.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[];
  static const ::google::protobuf::uint32 offsets[];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static void InitDefaultsImpl();
};
void AddDescriptors();
void InitDefaults();
}  // namespace protobuf_weights_2eproto

// ===================================================================

class Weights : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:yantk.Weights) */ {
 public:
  Weights();
  virtual ~Weights();

  Weights(const Weights& from);

  inline Weights& operator=(const Weights& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  Weights(Weights&& from) noexcept
    : Weights() {
    *this = ::std::move(from);
  }

  inline Weights& operator=(Weights&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Weights& default_instance();

  static inline const Weights* internal_default_instance() {
    return reinterpret_cast<const Weights*>(
               &_Weights_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(Weights* other);
  friend void swap(Weights& a, Weights& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline Weights* New() const PROTOBUF_FINAL { return New(NULL); }

  Weights* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Weights& from);
  void MergeFrom(const Weights& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Weights* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required float w_h_00 = 1;
  bool has_w_h_00() const;
  void clear_w_h_00();
  static const int kWH00FieldNumber = 1;
  float w_h_00() const;
  void set_w_h_00(float value);

  // required float w_h_01 = 2;
  bool has_w_h_01() const;
  void clear_w_h_01();
  static const int kWH01FieldNumber = 2;
  float w_h_01() const;
  void set_w_h_01(float value);

  // required float w_h_10 = 3;
  bool has_w_h_10() const;
  void clear_w_h_10();
  static const int kWH10FieldNumber = 3;
  float w_h_10() const;
  void set_w_h_10(float value);

  // required float w_h_11 = 4;
  bool has_w_h_11() const;
  void clear_w_h_11();
  static const int kWH11FieldNumber = 4;
  float w_h_11() const;
  void set_w_h_11(float value);

  // required float b_0 = 5;
  bool has_b_0() const;
  void clear_b_0();
  static const int kB0FieldNumber = 5;
  float b_0() const;
  void set_b_0(float value);

  // required float b_1 = 6;
  bool has_b_1() const;
  void clear_b_1();
  static const int kB1FieldNumber = 6;
  float b_1() const;
  void set_b_1(float value);

  // required float w_o_00 = 7;
  bool has_w_o_00() const;
  void clear_w_o_00();
  static const int kWO00FieldNumber = 7;
  float w_o_00() const;
  void set_w_o_00(float value);

  // required float w_o_01 = 8;
  bool has_w_o_01() const;
  void clear_w_o_01();
  static const int kWO01FieldNumber = 8;
  float w_o_01() const;
  void set_w_o_01(float value);

  // required float w_o_10 = 9;
  bool has_w_o_10() const;
  void clear_w_o_10();
  static const int kWO10FieldNumber = 9;
  float w_o_10() const;
  void set_w_o_10(float value);

  // required float w_o_11 = 10;
  bool has_w_o_11() const;
  void clear_w_o_11();
  static const int kWO11FieldNumber = 10;
  float w_o_11() const;
  void set_w_o_11(float value);

  // @@protoc_insertion_point(class_scope:yantk.Weights)
 private:
  void set_has_w_h_00();
  void clear_has_w_h_00();
  void set_has_w_h_01();
  void clear_has_w_h_01();
  void set_has_w_h_10();
  void clear_has_w_h_10();
  void set_has_w_h_11();
  void clear_has_w_h_11();
  void set_has_b_0();
  void clear_has_b_0();
  void set_has_b_1();
  void clear_has_b_1();
  void set_has_w_o_00();
  void clear_has_w_o_00();
  void set_has_w_o_01();
  void clear_has_w_o_01();
  void set_has_w_o_10();
  void clear_has_w_o_10();
  void set_has_w_o_11();
  void clear_has_w_o_11();

  // helper for ByteSizeLong()
  size_t RequiredFieldsByteSizeFallback() const;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  float w_h_00_;
  float w_h_01_;
  float w_h_10_;
  float w_h_11_;
  float b_0_;
  float b_1_;
  float w_o_00_;
  float w_o_01_;
  float w_o_10_;
  float w_o_11_;
  friend struct protobuf_weights_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class IterationWeights : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:yantk.IterationWeights) */ {
 public:
  IterationWeights();
  virtual ~IterationWeights();

  IterationWeights(const IterationWeights& from);

  inline IterationWeights& operator=(const IterationWeights& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  IterationWeights(IterationWeights&& from) noexcept
    : IterationWeights() {
    *this = ::std::move(from);
  }

  inline IterationWeights& operator=(IterationWeights&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const IterationWeights& default_instance();

  static inline const IterationWeights* internal_default_instance() {
    return reinterpret_cast<const IterationWeights*>(
               &_IterationWeights_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    1;

  void Swap(IterationWeights* other);
  friend void swap(IterationWeights& a, IterationWeights& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline IterationWeights* New() const PROTOBUF_FINAL { return New(NULL); }

  IterationWeights* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const IterationWeights& from);
  void MergeFrom(const IterationWeights& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(IterationWeights* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required .yantk.Weights weights = 2;
  bool has_weights() const;
  void clear_weights();
  static const int kWeightsFieldNumber = 2;
  const ::yantk::Weights& weights() const;
  ::yantk::Weights* mutable_weights();
  ::yantk::Weights* release_weights();
  void set_allocated_weights(::yantk::Weights* weights);

  // required int32 iteration = 1;
  bool has_iteration() const;
  void clear_iteration();
  static const int kIterationFieldNumber = 1;
  ::google::protobuf::int32 iteration() const;
  void set_iteration(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:yantk.IterationWeights)
 private:
  void set_has_iteration();
  void clear_has_iteration();
  void set_has_weights();
  void clear_has_weights();

  // helper for ByteSizeLong()
  size_t RequiredFieldsByteSizeFallback() const;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  ::yantk::Weights* weights_;
  ::google::protobuf::int32 iteration_;
  friend struct protobuf_weights_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class TrainingWeights : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:yantk.TrainingWeights) */ {
 public:
  TrainingWeights();
  virtual ~TrainingWeights();

  TrainingWeights(const TrainingWeights& from);

  inline TrainingWeights& operator=(const TrainingWeights& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  TrainingWeights(TrainingWeights&& from) noexcept
    : TrainingWeights() {
    *this = ::std::move(from);
  }

  inline TrainingWeights& operator=(TrainingWeights&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const TrainingWeights& default_instance();

  static inline const TrainingWeights* internal_default_instance() {
    return reinterpret_cast<const TrainingWeights*>(
               &_TrainingWeights_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    2;

  void Swap(TrainingWeights* other);
  friend void swap(TrainingWeights& a, TrainingWeights& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline TrainingWeights* New() const PROTOBUF_FINAL { return New(NULL); }

  TrainingWeights* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const TrainingWeights& from);
  void MergeFrom(const TrainingWeights& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(TrainingWeights* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .yantk.IterationWeights iteration_weights = 1;
  int iteration_weights_size() const;
  void clear_iteration_weights();
  static const int kIterationWeightsFieldNumber = 1;
  const ::yantk::IterationWeights& iteration_weights(int index) const;
  ::yantk::IterationWeights* mutable_iteration_weights(int index);
  ::yantk::IterationWeights* add_iteration_weights();
  ::google::protobuf::RepeatedPtrField< ::yantk::IterationWeights >*
      mutable_iteration_weights();
  const ::google::protobuf::RepeatedPtrField< ::yantk::IterationWeights >&
      iteration_weights() const;

  // @@protoc_insertion_point(class_scope:yantk.TrainingWeights)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  ::google::protobuf::RepeatedPtrField< ::yantk::IterationWeights > iteration_weights_;
  friend struct protobuf_weights_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Weights

// required float w_h_00 = 1;
inline bool Weights::has_w_h_00() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void Weights::set_has_w_h_00() {
  _has_bits_[0] |= 0x00000001u;
}
inline void Weights::clear_has_w_h_00() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void Weights::clear_w_h_00() {
  w_h_00_ = 0;
  clear_has_w_h_00();
}
inline float Weights::w_h_00() const {
  // @@protoc_insertion_point(field_get:yantk.Weights.w_h_00)
  return w_h_00_;
}
inline void Weights::set_w_h_00(float value) {
  set_has_w_h_00();
  w_h_00_ = value;
  // @@protoc_insertion_point(field_set:yantk.Weights.w_h_00)
}

// required float w_h_01 = 2;
inline bool Weights::has_w_h_01() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void Weights::set_has_w_h_01() {
  _has_bits_[0] |= 0x00000002u;
}
inline void Weights::clear_has_w_h_01() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void Weights::clear_w_h_01() {
  w_h_01_ = 0;
  clear_has_w_h_01();
}
inline float Weights::w_h_01() const {
  // @@protoc_insertion_point(field_get:yantk.Weights.w_h_01)
  return w_h_01_;
}
inline void Weights::set_w_h_01(float value) {
  set_has_w_h_01();
  w_h_01_ = value;
  // @@protoc_insertion_point(field_set:yantk.Weights.w_h_01)
}

// required float w_h_10 = 3;
inline bool Weights::has_w_h_10() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void Weights::set_has_w_h_10() {
  _has_bits_[0] |= 0x00000004u;
}
inline void Weights::clear_has_w_h_10() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void Weights::clear_w_h_10() {
  w_h_10_ = 0;
  clear_has_w_h_10();
}
inline float Weights::w_h_10() const {
  // @@protoc_insertion_point(field_get:yantk.Weights.w_h_10)
  return w_h_10_;
}
inline void Weights::set_w_h_10(float value) {
  set_has_w_h_10();
  w_h_10_ = value;
  // @@protoc_insertion_point(field_set:yantk.Weights.w_h_10)
}

// required float w_h_11 = 4;
inline bool Weights::has_w_h_11() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void Weights::set_has_w_h_11() {
  _has_bits_[0] |= 0x00000008u;
}
inline void Weights::clear_has_w_h_11() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void Weights::clear_w_h_11() {
  w_h_11_ = 0;
  clear_has_w_h_11();
}
inline float Weights::w_h_11() const {
  // @@protoc_insertion_point(field_get:yantk.Weights.w_h_11)
  return w_h_11_;
}
inline void Weights::set_w_h_11(float value) {
  set_has_w_h_11();
  w_h_11_ = value;
  // @@protoc_insertion_point(field_set:yantk.Weights.w_h_11)
}

// required float b_0 = 5;
inline bool Weights::has_b_0() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void Weights::set_has_b_0() {
  _has_bits_[0] |= 0x00000010u;
}
inline void Weights::clear_has_b_0() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void Weights::clear_b_0() {
  b_0_ = 0;
  clear_has_b_0();
}
inline float Weights::b_0() const {
  // @@protoc_insertion_point(field_get:yantk.Weights.b_0)
  return b_0_;
}
inline void Weights::set_b_0(float value) {
  set_has_b_0();
  b_0_ = value;
  // @@protoc_insertion_point(field_set:yantk.Weights.b_0)
}

// required float b_1 = 6;
inline bool Weights::has_b_1() const {
  return (_has_bits_[0] & 0x00000020u) != 0;
}
inline void Weights::set_has_b_1() {
  _has_bits_[0] |= 0x00000020u;
}
inline void Weights::clear_has_b_1() {
  _has_bits_[0] &= ~0x00000020u;
}
inline void Weights::clear_b_1() {
  b_1_ = 0;
  clear_has_b_1();
}
inline float Weights::b_1() const {
  // @@protoc_insertion_point(field_get:yantk.Weights.b_1)
  return b_1_;
}
inline void Weights::set_b_1(float value) {
  set_has_b_1();
  b_1_ = value;
  // @@protoc_insertion_point(field_set:yantk.Weights.b_1)
}

// required float w_o_00 = 7;
inline bool Weights::has_w_o_00() const {
  return (_has_bits_[0] & 0x00000040u) != 0;
}
inline void Weights::set_has_w_o_00() {
  _has_bits_[0] |= 0x00000040u;
}
inline void Weights::clear_has_w_o_00() {
  _has_bits_[0] &= ~0x00000040u;
}
inline void Weights::clear_w_o_00() {
  w_o_00_ = 0;
  clear_has_w_o_00();
}
inline float Weights::w_o_00() const {
  // @@protoc_insertion_point(field_get:yantk.Weights.w_o_00)
  return w_o_00_;
}
inline void Weights::set_w_o_00(float value) {
  set_has_w_o_00();
  w_o_00_ = value;
  // @@protoc_insertion_point(field_set:yantk.Weights.w_o_00)
}

// required float w_o_01 = 8;
inline bool Weights::has_w_o_01() const {
  return (_has_bits_[0] & 0x00000080u) != 0;
}
inline void Weights::set_has_w_o_01() {
  _has_bits_[0] |= 0x00000080u;
}
inline void Weights::clear_has_w_o_01() {
  _has_bits_[0] &= ~0x00000080u;
}
inline void Weights::clear_w_o_01() {
  w_o_01_ = 0;
  clear_has_w_o_01();
}
inline float Weights::w_o_01() const {
  // @@protoc_insertion_point(field_get:yantk.Weights.w_o_01)
  return w_o_01_;
}
inline void Weights::set_w_o_01(float value) {
  set_has_w_o_01();
  w_o_01_ = value;
  // @@protoc_insertion_point(field_set:yantk.Weights.w_o_01)
}

// required float w_o_10 = 9;
inline bool Weights::has_w_o_10() const {
  return (_has_bits_[0] & 0x00000100u) != 0;
}
inline void Weights::set_has_w_o_10() {
  _has_bits_[0] |= 0x00000100u;
}
inline void Weights::clear_has_w_o_10() {
  _has_bits_[0] &= ~0x00000100u;
}
inline void Weights::clear_w_o_10() {
  w_o_10_ = 0;
  clear_has_w_o_10();
}
inline float Weights::w_o_10() const {
  // @@protoc_insertion_point(field_get:yantk.Weights.w_o_10)
  return w_o_10_;
}
inline void Weights::set_w_o_10(float value) {
  set_has_w_o_10();
  w_o_10_ = value;
  // @@protoc_insertion_point(field_set:yantk.Weights.w_o_10)
}

// required float w_o_11 = 10;
inline bool Weights::has_w_o_11() const {
  return (_has_bits_[0] & 0x00000200u) != 0;
}
inline void Weights::set_has_w_o_11() {
  _has_bits_[0] |= 0x00000200u;
}
inline void Weights::clear_has_w_o_11() {
  _has_bits_[0] &= ~0x00000200u;
}
inline void Weights::clear_w_o_11() {
  w_o_11_ = 0;
  clear_has_w_o_11();
}
inline float Weights::w_o_11() const {
  // @@protoc_insertion_point(field_get:yantk.Weights.w_o_11)
  return w_o_11_;
}
inline void Weights::set_w_o_11(float value) {
  set_has_w_o_11();
  w_o_11_ = value;
  // @@protoc_insertion_point(field_set:yantk.Weights.w_o_11)
}

// -------------------------------------------------------------------

// IterationWeights

// required int32 iteration = 1;
inline bool IterationWeights::has_iteration() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void IterationWeights::set_has_iteration() {
  _has_bits_[0] |= 0x00000002u;
}
inline void IterationWeights::clear_has_iteration() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void IterationWeights::clear_iteration() {
  iteration_ = 0;
  clear_has_iteration();
}
inline ::google::protobuf::int32 IterationWeights::iteration() const {
  // @@protoc_insertion_point(field_get:yantk.IterationWeights.iteration)
  return iteration_;
}
inline void IterationWeights::set_iteration(::google::protobuf::int32 value) {
  set_has_iteration();
  iteration_ = value;
  // @@protoc_insertion_point(field_set:yantk.IterationWeights.iteration)
}

// required .yantk.Weights weights = 2;
inline bool IterationWeights::has_weights() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void IterationWeights::set_has_weights() {
  _has_bits_[0] |= 0x00000001u;
}
inline void IterationWeights::clear_has_weights() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void IterationWeights::clear_weights() {
  if (weights_ != NULL) weights_->::yantk::Weights::Clear();
  clear_has_weights();
}
inline const ::yantk::Weights& IterationWeights::weights() const {
  const ::yantk::Weights* p = weights_;
  // @@protoc_insertion_point(field_get:yantk.IterationWeights.weights)
  return p != NULL ? *p : *reinterpret_cast<const ::yantk::Weights*>(
      &::yantk::_Weights_default_instance_);
}
inline ::yantk::Weights* IterationWeights::mutable_weights() {
  set_has_weights();
  if (weights_ == NULL) {
    weights_ = new ::yantk::Weights;
  }
  // @@protoc_insertion_point(field_mutable:yantk.IterationWeights.weights)
  return weights_;
}
inline ::yantk::Weights* IterationWeights::release_weights() {
  // @@protoc_insertion_point(field_release:yantk.IterationWeights.weights)
  clear_has_weights();
  ::yantk::Weights* temp = weights_;
  weights_ = NULL;
  return temp;
}
inline void IterationWeights::set_allocated_weights(::yantk::Weights* weights) {
  delete weights_;
  weights_ = weights;
  if (weights) {
    set_has_weights();
  } else {
    clear_has_weights();
  }
  // @@protoc_insertion_point(field_set_allocated:yantk.IterationWeights.weights)
}

// -------------------------------------------------------------------

// TrainingWeights

// repeated .yantk.IterationWeights iteration_weights = 1;
inline int TrainingWeights::iteration_weights_size() const {
  return iteration_weights_.size();
}
inline void TrainingWeights::clear_iteration_weights() {
  iteration_weights_.Clear();
}
inline const ::yantk::IterationWeights& TrainingWeights::iteration_weights(int index) const {
  // @@protoc_insertion_point(field_get:yantk.TrainingWeights.iteration_weights)
  return iteration_weights_.Get(index);
}
inline ::yantk::IterationWeights* TrainingWeights::mutable_iteration_weights(int index) {
  // @@protoc_insertion_point(field_mutable:yantk.TrainingWeights.iteration_weights)
  return iteration_weights_.Mutable(index);
}
inline ::yantk::IterationWeights* TrainingWeights::add_iteration_weights() {
  // @@protoc_insertion_point(field_add:yantk.TrainingWeights.iteration_weights)
  return iteration_weights_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::yantk::IterationWeights >*
TrainingWeights::mutable_iteration_weights() {
  // @@protoc_insertion_point(field_mutable_list:yantk.TrainingWeights.iteration_weights)
  return &iteration_weights_;
}
inline const ::google::protobuf::RepeatedPtrField< ::yantk::IterationWeights >&
TrainingWeights::iteration_weights() const {
  // @@protoc_insertion_point(field_list:yantk.TrainingWeights.iteration_weights)
  return iteration_weights_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)


}  // namespace yantk

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_weights_2eproto__INCLUDED
