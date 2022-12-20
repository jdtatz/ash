#![recursion_limit = "256"]
#![warn(trivial_casts, trivial_numeric_casts)]

use heck::{ToShoutySnakeCase, ToSnakeCase, ToUpperCamelCase};
use itertools::Itertools;
use once_cell::sync::Lazy;
use proc_macro2::{Ident, Literal, TokenStream};
use quote::*;
use regex::Regex;
use std::fs::{self, File};
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Display,
    path::Path,
};
use vulkan_parse::*;

macro_rules! get_variant {
    ($variant:path) => {
        |enum_| match enum_ {
            $variant(inner) => Some(inner),
            _ => None,
        }
    };
    ($variant:path { $($member:ident),+ }) => {
        |enum_| match enum_ {
            $variant { $($member),+, .. } => Some(( $($member),+ )),
            _ => None,
        }
    };
}

const BACKWARDS_COMPATIBLE_ALIAS_COMMENT: &str = "Backwards-compatible alias containing a typo";

pub trait ExtensionExt {}

fn bin_op_tokens(op: &BinaryOp) -> TokenStream {
    match op {
        BinaryOp::Addition => quote!(+),
        BinaryOp::Subtraction => quote!(-),
        BinaryOp::Multiplication => quote!(*),
        BinaryOp::Division => quote!(/),
        BinaryOp::Remainder => quote!(%),
        BinaryOp::LeftShift => quote!(<<),
        BinaryOp::RightShift => quote!(>>),
        BinaryOp::BitwiseAnd => quote!(&),
        BinaryOp::BitwiseOr => quote!(|),
        BinaryOp::BitwiseXor => quote!(^),
        BinaryOp::LogicalAnd => quote!(&&),
        BinaryOp::LogicalOr => quote!(||),
    }
}

fn comp_op_tokens(op: &ComparisionOp) -> TokenStream {
    match op {
        ComparisionOp::LT => quote!(<),
        ComparisionOp::LTE => quote!(<=),
        ComparisionOp::Eq => quote!(==),
        ComparisionOp::NEq => quote!(!=),
        ComparisionOp::GTE => quote!(>=),
        ComparisionOp::GT => quote!(>),
    }
}

fn expression_tokens(
    expr: &Expression,
    identifier_renames: &BTreeMap<&str, Ident>,
    is_inner: bool,
) -> TokenStream {
    match expr {
        Expression::Identifier(ident) => {
            let i = identifier_renames
                .get(ident)
                .cloned()
                .unwrap_or_else(|| name_to_tokens(constant_name(ident)));
            quote!(#i)
        }
        Expression::Constant(vulkan_parse::Constant::Char(_)) => todo!(),
        Expression::Constant(vulkan_parse::Constant::Integer(i)) => {
            let lit = Literal::u64_unsuffixed(*i);
            quote!(#lit)
        }
        Expression::Constant(vulkan_parse::Constant::Float(f)) => {
            let lit = Literal::f64_unsuffixed(*f);
            quote!(#lit)
        }
        Expression::Literal(lit) => quote!(#lit),
        Expression::SizeOf(_) => todo!(),
        Expression::Unary(UnaryOp::Positive, v) => {
            let v = expression_tokens(v, identifier_renames, true);
            quote!(+#v)
        }
        Expression::Unary(UnaryOp::Negative, v) => {
            let v = expression_tokens(v, identifier_renames, true);
            quote!(-#v)
        }
        Expression::Unary(UnaryOp::BitwiseNegation | UnaryOp::LogicalNegation, v) => {
            let v = expression_tokens(v, identifier_renames, true);
            quote!(!#v)
        }
        Expression::Unary(UnaryOp::Cast(_), v) => {
            expression_tokens(v, identifier_renames, is_inner)
        }
        Expression::Unary(op, _) => todo!("{:?}", op),
        Expression::Binary(op, l, r) => {
            let op = bin_op_tokens(op);
            let l = expression_tokens(l, identifier_renames, true);
            let r = expression_tokens(r, identifier_renames, true);
            if is_inner {
                quote!((#l #op #r))
            } else {
                quote!(#l #op #r)
            }
        }
        Expression::Comparision(op, l, r) => {
            let op = comp_op_tokens(op);
            let l = expression_tokens(l, identifier_renames, true);
            let r = expression_tokens(r, identifier_renames, true);
            if is_inner {
                quote!((#l #op #r))
            } else {
                quote!(#l #op #r)
            }
        }
        Expression::Assignment(_, _, _) => todo!(),
        Expression::TernaryIfElse(_, _, _) => todo!(),
        Expression::FunctionCall(fn_expr, args) => {
            let f = expression_tokens(fn_expr, identifier_renames, true);
            let args = args
                .iter()
                .map(|e| expression_tokens(e, identifier_renames, false));
            if is_inner {
                quote!((#f(#(#args),*)))
            } else {
                quote!(#f(#(#args),*))
            }
        }
        Expression::Comma(_, _) => todo!(),
        Expression::Member(_, _, _) => todo!(),
        Expression::ArrayElement(_, _) => todo!(),
    }
}

fn khronos_link<S: Display + ?Sized>(name: &S) -> Literal {
    Literal::string(&format!(
        "<https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/{name}.html>",
        name = name
    ))
}

// FIXME read from vk.xml
fn is_opaque_type(ty: &str) -> bool {
    matches!(
        ty,
        "void"
            | "wl_display"
            | "wl_surface"
            | "Display"
            | "xcb_connection_t"
            | "ANativeWindow"
            | "AHardwareBuffer"
            | "CAMetalLayer"
            | "IDirectFB"
            | "IDirectFBSurface"
    )
}

pub trait ConstantExt {
    fn name(&self) -> &str;
    fn extension_constant(
        &self,
        enum_name: &str,
        extension_number: Option<i64>,
        vendors: &[&str],
    ) -> Constant;
    fn constant(&self, enum_name: &str, vendors: &[&str]) -> Constant {
        self.extension_constant(enum_name, None, vendors)
    }
    fn variant_ident(&self, enum_name: &str, vendors: &[&str]) -> Ident {
        variant_ident(enum_name, self.name(), vendors)
    }
    fn notation(&self) -> Option<&str>;
    fn formatted_notation(&self) -> Option<Cow<'_, str>> {
        static DOC_LINK: Lazy<Regex> = Lazy::new(|| Regex::new(r#"<<([\w-]+)>>"#).unwrap());
        self.notation().map(|n| {
            DOC_LINK.replace(
                n,
                "<https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#${1}>",
            )
        })
    }
    fn is_alias(&self) -> bool {
        false
    }
    fn doc_attribute(&self) -> TokenStream {
        assert_ne!(
            self.notation(),
            Some(BACKWARDS_COMPATIBLE_ALIAS_COMMENT),
            "Backwards-compatible constants should not be emitted"
        );
        match self.formatted_notation() {
            // FIXME: Only used for `VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE_KHR`
            Some(n) if n.starts_with("Alias") => quote!(#[deprecated = #n]),
            Some(n) => quote!(#[doc = #n]),
            None => quote!(),
        }
    }
}

impl ConstantExt for ConstantEnum<'_> {
    fn name(&self) -> &str {
        &self.name
    }
    fn extension_constant(
        &self,
        _enum_name: &str,
        _extension_number: Option<i64>,
        _vendors: &[&str],
    ) -> Constant {
        Constant::Value(self.value.clone())
    }
    fn notation(&self) -> Option<&str> {
        self.comment.as_deref()
    }
    fn is_alias(&self) -> bool {
        false
    }
}

impl ConstantExt for ValueEnum<'_> {
    fn name(&self) -> &str {
        &self.name
    }
    fn extension_constant(
        &self,
        _enum_name: &str,
        _extension_number: Option<i64>,
        _vendors: &[&str],
    ) -> Constant {
        Constant::Offset(self.value)
    }
    fn notation(&self) -> Option<&str> {
        self.comment.as_deref()
    }
    fn is_alias(&self) -> bool {
        false
    }
}

impl ConstantExt for BitPosEnum<'_> {
    fn name(&self) -> &str {
        &self.name
    }
    fn extension_constant(
        &self,
        _enum_name: &str,
        _extension_number: Option<i64>,
        _vendors: &[&str],
    ) -> Constant {
        Constant::BitPos(self.bitpos.into())
    }
    fn notation(&self) -> Option<&str> {
        self.comment.as_deref()
    }
    fn is_alias(&self) -> bool {
        false
    }
}

impl ConstantExt for BitmaskEnum<'_> {
    fn name(&self) -> &str {
        match self {
            BitmaskEnum::Value(v) => v.name(),
            BitmaskEnum::BitPos(b) => b.name(),
        }
    }
    fn extension_constant(
        &self,
        enum_name: &str,
        extension_number: Option<i64>,
        vendors: &[&str],
    ) -> Constant {
        match self {
            BitmaskEnum::Value(v) => v.extension_constant(enum_name, extension_number, vendors),
            BitmaskEnum::BitPos(b) => b.extension_constant(enum_name, extension_number, vendors),
        }
    }
    fn notation(&self) -> Option<&str> {
        match self {
            BitmaskEnum::Value(v) => v.notation(),
            BitmaskEnum::BitPos(b) => b.notation(),
        }
    }
    fn is_alias(&self) -> bool {
        match self {
            BitmaskEnum::Value(v) => v.is_alias(),
            BitmaskEnum::BitPos(b) => b.is_alias(),
        }
    }
}

impl ConstantExt for Alias<'_> {
    fn name(&self) -> &str {
        &self.name
    }
    fn extension_constant(
        &self,
        enum_name: &str,
        _extension_number: Option<i64>,
        vendors: &[&str],
    ) -> Constant {
        Constant::Alias(variant_ident(enum_name, &self.alias, vendors))
    }
    fn notation(&self) -> Option<&str> {
        self.comment.as_deref()
    }
    fn is_alias(&self) -> bool {
        true
    }
}

impl<T: ConstantExt> ConstantExt for DefinitionOrAlias<'_, T> {
    fn name(&self) -> &str {
        match self {
            Self::Alias(a) => a.name(),
            Self::Definition(b) => b.name(),
        }
    }
    fn extension_constant(
        &self,
        enum_name: &str,
        extension_number: Option<i64>,
        vendors: &[&str],
    ) -> Constant {
        match self {
            Self::Alias(a) => a.extension_constant(enum_name, extension_number, vendors),
            Self::Definition(b) => b.extension_constant(enum_name, extension_number, vendors),
        }
    }
    fn notation(&self) -> Option<&str> {
        match self {
            Self::Alias(a) => a.notation(),
            Self::Definition(b) => b.notation(),
        }
    }
    fn is_alias(&self) -> bool {
        match self {
            Self::Alias(_) => true,
            Self::Definition(b) => b.is_alias(),
        }
    }
}

impl ConstantExt for RequireEnum<'_> {
    fn name(&self) -> &str {
        self.name.as_deref().unwrap()
    }
    fn extension_constant(
        &self,
        enum_name: &str,
        extension_number: Option<i64>,
        vendors: &[&str],
    ) -> Constant {
        match self.value.as_ref().unwrap() {
            RequireValueEnum::Value(value) => Constant::Value(value.clone()),
            RequireValueEnum::Alias(alias) => {
                let key =
                    variant_ident(self.extends.as_deref().unwrap_or(enum_name), alias, vendors);
                Constant::Alias(key)
            }
            RequireValueEnum::Offset {
                extnumber,
                offset,
                direction,
            } => {
                let ext_base = 1_000_000_000;
                let ext_block_size = 1000;
                let extnumber = extnumber.map_or_else(|| extension_number.unwrap(), |i| i as i64);
                let value = ext_base + (extnumber - 1) * ext_block_size + (*offset as i64);
                let value = if *direction != Some(OffsetDirection::Negative) {
                    value
                } else {
                    -value
                };
                Constant::Offset(value)
            }
            RequireValueEnum::Bitpos(bitpos) => Constant::BitPos(*bitpos as _),
        }
    }
    fn notation(&self) -> Option<&str> {
        self.comment.as_deref()
    }
    fn is_alias(&self) -> bool {
        matches!(self.value, Some(RequireValueEnum::Alias(_)))
    }
}

#[derive(Clone, Debug)]
pub enum Constant<'a> {
    Value(Expression<'a>),
    Size(usize),
    BitPos(u32),
    Offset(i64),
    Alias(Ident),
}

impl<'a> quote::ToTokens for Constant<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Constant::Value(v) => expression_tokens(v, &BTreeMap::new(), false).to_tokens(tokens),
            Constant::Offset(size) => {
                // Literal::i64_unsuffixed(*size).to_tokens(tokens)
                let number = interleave_number('_', 3, size.to_string().as_str());
                number.parse::<Literal>().unwrap().to_tokens(tokens);
            }
            Constant::Size(size) => Literal::usize_unsuffixed(*size).to_tokens(tokens),
            Constant::BitPos(pos) => {
                let value = 1u64 << pos;
                let bit_string = format!("{:b}", value);
                let bit_string = interleave_number('_', 4, &bit_string);
                format!("0b{:}", bit_string)
                    .parse::<Literal>()
                    .unwrap()
                    .to_tokens(tokens);
            }
            Constant::Alias(ref value) => tokens.extend(quote!(Self::#value)),
        }
    }
}

// Interleaves a number, for example 100000 => 100_000. Mostly used to make clippy happy
fn interleave_number(symbol: char, count: usize, n: &str) -> String {
    let number: String = n
        .chars()
        .rev()
        .enumerate()
        .fold(String::new(), |mut acc, (idx, next)| {
            if idx != 0 && idx % count == 0 {
                acc.push(symbol);
            }
            acc.push(next);
            acc
        });
    number.chars().rev().collect()
}

pub trait FeatureExt {
    fn version_string(&self) -> String;
    fn is_version(&self, major: u32, minor: u32) -> bool;
}
impl FeatureExt for Feature<'_> {
    fn is_version(&self, major: u32, minor: u32) -> bool {
        major == self.number.major && minor == self.number.minor
    }
    fn version_string(&self) -> String {
        format!("{}_{}", self.number.major, self.number.minor)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum FunctionType {
    Static,
    Entry,
    Instance,
    Device,
}
pub trait CommandExt {
    /// Returns the ident in snake_case and without the 'vk' prefix.
    fn function_type(&self) -> FunctionType;
    ///
    /// Returns true if the command is a device level command. This is indicated by
    /// the type of the first parameter.
    fn command_ident(&self) -> Ident;
}

impl CommandExt for Command<'_> {
    fn command_ident(&self) -> Ident {
        format_ident!(
            "{}",
            self.proto.name.strip_prefix("vk").unwrap().to_snake_case()
        )
    }

    // FIXME derive this from dispatchable handles
    fn function_type(&self) -> FunctionType {
        let is_first_param_device = self.params.get(0).map_or(false, |field| {
            matches!(
                field.type_name.as_identifier(),
                "VkDevice" | "VkCommandBuffer" | "VkQueue"
            )
        });
        match self.proto.name.as_ref() {
            "vkGetInstanceProcAddr" => FunctionType::Static,
            "vkCreateInstance"
            | "vkEnumerateInstanceLayerProperties"
            | "vkEnumerateInstanceExtensionProperties"
            | "vkEnumerateInstanceVersion" => FunctionType::Entry,
            // This is actually not a device level function
            "vkGetDeviceProcAddr" => FunctionType::Instance,
            _ => {
                if is_first_param_device {
                    FunctionType::Device
                } else {
                    FunctionType::Instance
                }
            }
        }
    }
}

pub trait FieldExt {
    /// Returns the name of the parameter that doesn't clash with Rusts reserved
    /// keywords
    fn param_ident(&self) -> Ident;

    /// The inner type of this field, with one level of pointers removed
    fn inner_type_tokens(&self, lifetime: Option<TokenStream>) -> TokenStream;

    /// Returns reference-types wrapped in their safe variant. (Dynamic) arrays become
    /// slices, pointers become Rust references.
    fn safe_type_tokens(&self, lifetime: TokenStream) -> TokenStream;

    /// Returns the basetype ident and removes the 'Vk' prefix. When `is_ffi_param` is `true`
    /// array types (e.g. `[f32; 3]`) will be converted to pointer types (e.g. `&[f32; 3]`),
    /// which is needed for `C` function parameters. Set to `false` for struct definitions.
    fn type_tokens(&self, is_ffi_param: bool) -> TokenStream;

    /// Whether this is C's `void` type (not to be mistaken with a void _pointer_!)
    fn is_void(&self) -> bool;

    /// Exceptions for pointers to static-sized arrays,
    /// `vk.xml` does not annotate this.
    fn is_pointer_to_static_sized_array(&self) -> bool;
}

pub trait ToTokens {
    fn to_tokens(&self) -> TokenStream;
    /// Returns the topmost pointer as safe reference
    fn to_safe_tokens(&self, lifetime: TokenStream) -> TokenStream;
}

fn name_to_tokens(type_name: &str) -> Ident {
    let new_name = match type_name {
        "uint8_t" => "u8",
        "uint16_t" => "u16",
        "uint32_t" => "u32",
        "uint64_t" => "u64",
        "int8_t" => "i8",
        "int16_t" => "i16",
        "int32_t" => "i32",
        "int64_t" => "i64",
        // FIXME change to c_size_t once it's stable and in MSRV
        // see: https://github.com/rust-lang/rust/issues/88345
        "size_t" => "usize",
        "int" => "c_int",
        "void" => "c_void",
        "char" => "c_char",
        "float" => "f32",
        "double" => "f64",
        "long" => "c_ulong",
        _ => type_name.strip_prefix("Vk").unwrap_or(type_name),
    };
    let new_name = new_name.replace("FlagBits", "Flags");
    format_ident!("{}", new_name.as_str())
}

struct WrappedPointerKind<'a> {
    pointer_kind: Option<&'a PointerKind>,
    is_const: bool,
}

impl<'a> ToTokens for WrappedPointerKind<'a> {
    fn to_tokens(&self) -> TokenStream {
        let outer_pointer = self.pointer_kind.and_then(|pk| match pk {
            PointerKind::Single => None,
            PointerKind::Double { .. } if self.is_const => Some(quote!(*const)),
            PointerKind::Double { .. } => Some(quote!(*mut)),
        });
        let inner_pointer = self.pointer_kind.map(|pk| match pk {
            PointerKind::Single if self.is_const => quote!(*const),
            PointerKind::Single => quote!(*mut),
            PointerKind::Double {
                inner_is_const: true,
            } => quote!(*const),
            PointerKind::Double {
                inner_is_const: false,
            } => quote!(*mut),
        });
        quote!(#outer_pointer #inner_pointer)
    }

    fn to_safe_tokens(&self, lifetime: TokenStream) -> TokenStream {
        let outer_pointer = self.pointer_kind.and_then(|pk| match pk {
            PointerKind::Single => None,
            PointerKind::Double { .. } if self.is_const => Some(quote!(&#lifetime)),
            PointerKind::Double { .. } => Some(quote!(&#lifetime mut)),
        });
        let inner_pointer = self.pointer_kind.map(|pk| match pk {
            PointerKind::Single if self.is_const => quote!(&#lifetime),
            PointerKind::Single => quote!(&#lifetime mut),
            PointerKind::Double {
                inner_is_const: true,
            } => quote!(*const),
            PointerKind::Double {
                inner_is_const: false,
            } => quote!(*mut),
        });
        quote!(#outer_pointer #inner_pointer)
    }
}

impl FieldExt for FieldLike<'_> {
    fn param_ident(&self) -> Ident {
        let name = self.name.as_ref();
        let name_corrected = match name {
            "type" => "ty",
            _ => name,
        };
        format_ident!("{}", name_corrected.to_snake_case().as_str())
    }

    fn inner_type_tokens(&self, lifetime: Option<TokenStream>) -> TokenStream {
        assert!(!self.is_void());
        let ty = name_to_tokens(self.type_name.as_identifier());

        // If the nested "dynamic array" has length 1, it's just a pointer which we convert to a safe borrow for convenience
        if let Some(DynamicShapeKind::Double(_, DynamicLength::Static(n))) = self.dynamic_shape {
            if n.get() == 1 {
                let lifetime = lifetime.unwrap();
                match self.pointer_kind {
                    Some(PointerKind::Double {
                        inner_is_const: true,
                    }) => quote!(&#lifetime #ty),
                    Some(PointerKind::Double {
                        inner_is_const: false,
                    }) => quote!(&#lifetime mut #ty),
                    _ => quote!(#ty),
                }
            } else {
                todo!()
            }
        } else {
            match self.pointer_kind {
                Some(PointerKind::Double {
                    inner_is_const: true,
                }) => quote!(*const #ty),
                Some(PointerKind::Double {
                    inner_is_const: false,
                }) => quote!(*mut #ty),
                _ => quote!(#ty),
            }
        }
    }

    fn safe_type_tokens(&self, lifetime: TokenStream) -> TokenStream {
        assert!(!self.is_void());
        if self.dynamic_shape.is_some() {
            let ty = self.inner_type_tokens(Some(lifetime));
            quote!([#ty])
        } else if self.array_shape().is_some() {
            // The outer type fn type_tokens() returns is [], which fits our "safe" prescription
            self.type_tokens(false)
        } else {
            let ty = name_to_tokens(self.type_name.as_identifier());
            let ptr = WrappedPointerKind {
                pointer_kind: self.pointer_kind.as_ref(),
                is_const: self.is_const,
            }
            .to_safe_tokens(lifetime);
            quote!(#ptr #ty)
        }
    }

    fn type_tokens(&self, is_ffi_param: bool) -> TokenStream {
        assert!(!self.is_void());
        let ty = name_to_tokens(self.type_name.as_identifier());
        let ptr = WrappedPointerKind {
            pointer_kind: self.pointer_kind.as_ref(),
            is_const: self.is_const,
        }
        .to_tokens();

        match &self.array_shape() {
            Some(shape) => {
                // Make sure we also rename the constant, that is
                // used inside the static array
                let shape = shape.iter().rev().fold(quote!(#ty), |ty, size| match size {
                    ArrayLength::Static(n) => {
                        let n = n.get();
                        let n = Literal::usize_unsuffixed(n as usize);
                        quote!([#ty; #n])
                    }
                    ArrayLength::Constant(size) => {
                        let size = format_ident!("{}", constant_name(size));
                        quote!([#ty; #size])
                    }
                });
                // arrays in c are always passed as a pointer
                if is_ffi_param {
                    quote!(*const #shape)
                } else {
                    quote!(#ptr #shape)
                }
            }
            _ => match &self.dynamic_shape {
                Some(DynamicShapeKind::Expression { c_expr, .. })
                    if self.is_pointer_to_static_sized_array() =>
                {
                    let size = expression_tokens(c_expr, &BTreeMap::new(), false);
                    quote!(#ptr [#ty; #size])
                }
                _ => quote!(#ptr #ty),
            },
        }
    }

    fn is_void(&self) -> bool {
        self.type_name == TypeSpecifier::Void && self.pointer_kind.is_none()
    }

    fn is_pointer_to_static_sized_array(&self) -> bool {
        self.dynamic_shape.is_some() && self.name == ("pVersionData")
    }
}

pub type CommandMap<'a, 's> = HashMap<&'s str, &'a Command<'a>>;

fn generate_function_pointers<'s, 'a: 's>(
    ident: &Ident,
    commands: &[&'s Command<'a>],
    aliases: &HashMap<&str, &str>,
    fn_cache: &mut HashSet<&'s str>,
) -> TokenStream {
    // Commands can have duplicates inside them because they are declared per features. But we only
    // really want to generate one function pointer.
    let commands = commands
        .iter()
        .unique_by(|cmd| cmd.proto.name)
        .collect::<Vec<_>>();

    struct CommandDefn {
        type_needs_defining: bool,
        type_name: Ident,
        function_name_c: String,
        function_name_rust: Ident,
        parameters: TokenStream,
        parameters_unused: TokenStream,
        returns: TokenStream,
    }

    let commands = commands
        .iter()
        .map(move |cmd| {
            let type_name = format_ident!("PFN_{}", cmd.proto.name);

            let function_name_c = if let Some(alias_name) = aliases.get(cmd.proto.name) {
                alias_name.to_string()
            } else {
                cmd.proto.name.to_string()
            };
            let function_name_rust = format_ident!(
                "{}",
                function_name_c
                    .strip_prefix("vk")
                    .unwrap()
                    .to_snake_case()
                    .as_str()
            );

            let params: Vec<_> = cmd
                .params
                .iter()
                .map(|field| {
                    let name = field.param_ident();
                    // FIXME
                    let ty = if let Some(valid_structs) = &field.valid_structs {
                        assert_eq!(valid_structs.len(), 1);
                        let ident = name_to_tokens(valid_structs[0].as_ref());
                        quote!(*mut #ident)
                    } else {
                        field.type_tokens(true)
                    };
                    (name, ty)
                })
                .collect();

            let params_iter = params
                .iter()
                .map(|(param_name, param_ty)| quote!(#param_name: #param_ty));
            let parameters = quote!(#(#params_iter,)*);

            let params_iter = params.iter().map(|(param_name, param_ty)| {
                let unused_name = format_ident!("_{}", param_name);
                quote!(#unused_name: #param_ty)
            });
            let parameters_unused = quote!(#(#params_iter,)*);

            CommandDefn {
                // PFN function pointers are global and can not have duplicates.
                // This can happen because there are aliases to commands
                type_needs_defining: fn_cache.insert(cmd.proto.name.as_ref()),
                type_name,
                function_name_c,
                function_name_rust,
                parameters,
                parameters_unused,
                returns: if cmd.proto.is_void() {
                    quote!()
                } else {
                    let ret_ty_tokens = cmd.proto.type_tokens(true);
                    quote!(-> #ret_ty_tokens)
                },
            }
        })
        .collect::<Vec<_>>();

    struct CommandToType<'a>(&'a CommandDefn);
    impl<'a> quote::ToTokens for CommandToType<'a> {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let type_name = &self.0.type_name;
            let parameters = &self.0.parameters;
            let returns = &self.0.returns;
            quote!(
                #[allow(non_camel_case_types)]
                pub type #type_name = unsafe extern "system" fn(#parameters) #returns;
            )
            .to_tokens(tokens)
        }
    }

    struct CommandToMember<'a>(&'a CommandDefn);
    impl<'a> quote::ToTokens for CommandToMember<'a> {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let type_name = &self.0.type_name;
            let type_name = if self.0.type_needs_defining {
                // Type is defined in local scope
                quote!(#type_name)
            } else {
                // Type is usually defined in another module
                quote!(crate::vk::#type_name)
            };
            let function_name_rust = &self.0.function_name_rust;
            quote!(pub #function_name_rust: #type_name).to_tokens(tokens)
        }
    }

    struct CommandToLoader<'a>(&'a CommandDefn);
    impl<'a> quote::ToTokens for CommandToLoader<'a> {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let function_name_rust = &self.0.function_name_rust;
            let parameters_unused = &self.0.parameters_unused;
            let returns = &self.0.returns;

            let byte_function_name =
                Literal::byte_string(format!("{}\0", self.0.function_name_c).as_bytes());

            quote!(
                #function_name_rust: unsafe {
                    unsafe extern "system" fn #function_name_rust (#parameters_unused) #returns {
                        panic!(concat!("Unable to load ", stringify!(#function_name_rust)))
                    }
                    let cname = ::std::ffi::CStr::from_bytes_with_nul_unchecked(#byte_function_name);
                    let val = _f(cname);
                    if val.is_null() {
                        #function_name_rust
                    } else {
                        ::std::mem::transmute(val)
                    }
                }
            )
            .to_tokens(tokens)
        }
    }

    let pfn_typedefs = commands
        .iter()
        .filter(|pfn| pfn.type_needs_defining)
        .map(CommandToType);
    let members = commands.iter().map(CommandToMember);
    let loaders = commands.iter().map(CommandToLoader);

    quote! {
        #(#pfn_typedefs)*

        #[derive(Clone)]
        pub struct #ident {
            #(#members,)*
        }

        unsafe impl Send for #ident {}
        unsafe impl Sync for #ident {}

        impl #ident {
            pub fn load<F>(mut _f: F) -> Self
                where F: FnMut(&::std::ffi::CStr) -> *const c_void
            {
                Self {
                    #(#loaders,)*
                }
            }
        }
    }
}
pub struct ExtensionConstant<'a> {
    pub name: &'a str,
    pub constant: Constant<'a>,
    pub notation: Option<&'a str>,
}
impl<'a> ConstantExt for ExtensionConstant<'a> {
    fn name(&self) -> &str {
        &self.name
    }
    fn extension_constant(
        &self,
        _enum_name: &str,
        _extension_number: Option<i64>,
        _vendors: &[&str],
    ) -> Constant {
        self.constant.clone()
    }
    fn notation(&self) -> Option<&str> {
        self.notation
    }
}

pub fn generate_extension_constants<'s, 'a: 's>(
    extension_name: &str,
    extension_number: i64,
    extension_items: &'s [Require<'a>],
    const_cache: &mut HashSet<&'s str>,
    const_values: &mut BTreeMap<Ident, ConstantTypeInfo>,
    vendors: &[&str],
) -> TokenStream {
    let items = extension_items.iter().flat_map(|r| r.values.values());

    let mut extended_enums = BTreeMap::<&str, Vec<ExtensionConstant>>::new();

    for item in items {
        if let RequireValue::Enum(
            req @ RequireEnum {
                name: Some(name),
                extends: Some(extends),
                comment,
                ..
            },
        ) = item
        {
            if !const_cache.insert(name.as_ref()) {
                continue;
            }

            if comment.as_deref() == Some(BACKWARDS_COMPATIBLE_ALIAS_COMMENT) {
                continue;
            }

            let constant = req.extension_constant(extends, Some(extension_number), vendors);
            let ext_constant = ExtensionConstant {
                name,
                constant,
                notation: comment.as_deref(),
            };
            let ext_const_ident = ext_constant.variant_ident(extends, vendors);
            let is_alias = if let Constant::Alias(alias_ident) = &ext_constant.constant {
                if alias_ident == &ext_const_ident {
                    continue;
                }
                true
            } else {
                false
            };
            let ident = name_to_tokens(extends);
            const_values
                .get_mut(&ident)
                .unwrap()
                .values
                .push(ConstantMatchInfo {
                    ident: ext_const_ident,
                    is_alias,
                });

            extended_enums
                .entry(extends)
                .or_default()
                .push(ext_constant);
        }
    }

    let enum_tokens = extended_enums.iter().map(|(extends, constants)| {
        let ident = name_to_tokens(extends);
        let doc_string = format!("Generated from '{}'", extension_name);
        let impl_block = bitflags_impl_block(&ident, extends, constants.as_slice(), vendors);
        quote! {
            #[doc = #doc_string]
            #impl_block
        }
    });
    quote!(#(#enum_tokens)*)
}
pub fn generate_extension_commands<'s, 'a: 's>(
    extension_name: &str,
    items: &'s [Require<'a>],
    cmd_map: &CommandMap<'a, 's>,
    cmd_aliases: &HashMap<&str, &str>,
    fn_cache: &mut HashSet<&'s str>,
) -> TokenStream {
    let mut commands = Vec::new();
    let mut aliases = HashMap::new();
    let names = items
        .iter()
        .flat_map(|r| r.values.values())
        .filter_map(get_variant!(RequireValue::Command { name }));
    for name in names {
        if let Some(cmd) = cmd_map.get(name).copied() {
            commands.push(cmd);
        } else if let Some(cmd) = cmd_aliases
            .get(name)
            .and_then(|alias_name| cmd_map.get(alias_name).copied())
        {
            aliases.insert(cmd.proto.name.as_ref(), name.as_ref());
            commands.push(cmd);
        }
    }

    let upper_ext_name = extension_name.to_upper_camel_case();
    let ident = format_ident!(
        "{}Fn",
        upper_ext_name.strip_prefix("Vk").unwrap_or(&upper_ext_name)
    );
    let fp = generate_function_pointers(&ident, &commands, &aliases, fn_cache);

    let spec_version = items
        .iter()
        .flat_map(|r| r.values.values())
        .filter_map(get_variant!(RequireValue::Enum))
        .find(|e| {
            e.name
                .as_deref()
                .map_or(false, |n| n.contains("SPEC_VERSION"))
        })
        .and_then(|e| {
            if let Some(RequireValueEnum::Value(Expression::Constant(
                vulkan_parse::Constant::Integer(i),
            ))) = &e.value
            {
                let v = *i as u32;
                Some(quote!(pub const SPEC_VERSION: u32 = #v;))
            } else {
                unreachable!()
            }
        });

    let byte_name_ident = Literal::byte_string(format!("{}\0", extension_name).as_bytes());
    let extension_cstr = quote! {
        impl #ident {
            #[inline]
            pub const fn name() -> &'static ::std::ffi::CStr {
                unsafe { ::std::ffi::CStr::from_bytes_with_nul_unchecked(#byte_name_ident) }
            }
            #spec_version
        }
    };
    quote! {
        #extension_cstr
        #fp
    }
}
pub fn generate_extension<'s, 'a: 's>(
    extension: &'s Extension<'a>,
    cmd_map: &CommandMap<'a, 's>,
    const_cache: &mut HashSet<&'s str>,
    const_values: &mut BTreeMap<Ident, ConstantTypeInfo>,
    cmd_aliases: &HashMap<&str, &str>,
    fn_cache: &mut HashSet<&'s str>,
    vendors: &[&str],
) -> Option<TokenStream> {
    // Okay this is a little bit odd. We need to generate all extensions, even disabled ones,
    // because otherwise some StructureTypes won't get generated. But we don't generate extensions
    // that are reserved
    if extension.name.contains("RESERVED") {
        return None;
    }
    let extension_tokens = generate_extension_constants(
        &extension.name,
        extension.number.into(),
        &extension.requires,
        const_cache,
        const_values,
        vendors,
    );
    let fp = generate_extension_commands(
        &extension.name,
        &extension.requires,
        cmd_map,
        cmd_aliases,
        fn_cache,
    );
    let q = quote! {
        #fp
        #extension_tokens
    };
    Some(q)
}
pub fn generate_define<'s>(
    define: &'s MacroDefine,
    identifier_renames: &mut BTreeMap<&'s str, Ident>,
) -> TokenStream {
    let name = constant_name(&define.name);
    let ident = format_ident!("{}", name);

    let link = khronos_link(&define.name);

    let deprecated = define
        .deprecation_comment
        .as_ref()
        .map(|c| c.trim())
        .map(|comment| quote!(#[deprecated = #comment]));

    if name == "NULL_HANDLE" || define.is_disabled {
        quote!()
    } else if let Some(value) = define.value.as_expr() {
        let v = expression_tokens(value.as_ref(), identifier_renames, false);
        quote!(
            #deprecated
            #[doc = #link]
            pub const #ident: u32 = #v;
        )
    } else {
        match &define.value {
            MacroDefineValue::FunctionDefine { params, expression }
                if define.name.contains("VERSION") =>
            {
                let c_expr: Expression = (&**expression).try_into().unwrap();
                let c_expr = expression_tokens(&c_expr, identifier_renames, false);

                let params = params
                    .iter()
                    .map(|param| format_ident!("{}", param))
                    .map(|i| quote!(#i: u32));
                let ident = format_ident!("{}", name.to_lowercase());
                let code = quote!(pub const fn #ident(#(#params),*) -> u32 { #c_expr });

                identifier_renames.insert(define.name.as_ref(), ident);

                quote! {
                    #deprecated
                    #[doc = #link]
                    #code
                }
            }
            _ => quote!(),
        }
    }
}

pub fn generate_typedef(typedef: &FieldLike) -> TokenStream {
    let typedef_name = name_to_tokens(&typedef.name);
    let typedef_ty = typedef.type_tokens(true);
    let khronos_link = khronos_link(&typedef.name);
    quote! {
        #[doc = #khronos_link]
        pub type #typedef_name = #typedef_ty;
    }
}
pub fn generate_bitmask(
    name: &str,
    is_64bit: bool,
    bitflags_cache: &mut HashSet<Ident>,
    const_values: &mut BTreeMap<Ident, ConstantTypeInfo>,
) -> Option<TokenStream> {
    let ident = name_to_tokens(name);
    if !bitflags_cache.insert(ident.clone()) {
        return None;
    };
    const_values.insert(ident.clone(), Default::default());
    let khronos_link = khronos_link(name);
    let type_name = if is_64bit { "VkFlags64" } else { "VkFlags" };
    let type_ = name_to_tokens(type_name);
    Some(quote! {
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[doc = #khronos_link]
        pub struct #ident(pub(crate) #type_);
        vk_bitflags_wrapped!(#ident, #type_);
    })
}

pub enum EnumType {
    Bitflags(TokenStream),
    Enum(TokenStream),
}

trait MaybeStrip {
    fn maybe_strip_prefix<'a>(&'a self, prefix: &str) -> &'a Self;
    fn maybe_strip_suffix<'a>(&'a self, suffix: &str) -> &'a Self;
}

impl MaybeStrip for str {
    fn maybe_strip_prefix<'a>(&'a self, prefix: &str) -> &'a Self {
        self.strip_prefix(prefix).unwrap_or(self)
    }

    fn maybe_strip_suffix<'a>(&'a self, suffix: &str) -> &'a Self {
        self.strip_suffix(suffix).unwrap_or(self)
    }
}

static TRAILING_NUMBER: Lazy<Regex> = Lazy::new(|| Regex::new("(\\d+)$").unwrap());

pub fn variant_ident(enum_name: &str, variant_name: &str, vendors: &[&str]) -> Ident {
    let variant_name = variant_name.to_uppercase();
    let name = enum_name.replace("FlagBits", "");
    let struct_name = name.to_shouty_snake_case();
    let vendor = vendors
        .iter()
        .find(|&vendor| struct_name.ends_with(vendor))
        .cloned()
        .unwrap_or("");
    let struct_name = struct_name
        .strip_suffix(vendor)
        .unwrap()
        .maybe_strip_suffix("_");
    let struct_name = TRAILING_NUMBER.replace(struct_name, "_$1");
    let variant_name = variant_name
        .maybe_strip_suffix(vendor)
        .maybe_strip_suffix("_");

    let new_variant_name = variant_name
        .strip_prefix(struct_name.as_ref())
        .unwrap_or_else(|| {
            if enum_name == "VkResult" {
                variant_name.strip_prefix("VK").unwrap()
            } else {
                panic!(
                    "Failed to strip {} prefix from enum variant {}",
                    struct_name, variant_name
                )
            }
        });

    // Both of the above strip_prefix leave a leading `_`:
    let new_variant_name = new_variant_name.strip_prefix('_').unwrap();
    // Replace _BIT anywhere in the string, also works when there's a trailing
    // vendor extension in the variant name that's not in the enum/type name:
    let new_variant_name = new_variant_name.replace("_BIT", "");
    let is_digit = new_variant_name
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false);
    if is_digit {
        format_ident!("TYPE_{}", new_variant_name)
    } else {
        format_ident!("{}", new_variant_name)
    }
}

pub fn bitflags_impl_block<'c, C: 'c + ConstantExt, I: IntoIterator<Item = &'c C>>(
    ident: &Ident,
    enum_name: &str,
    constants: I,
    vendors: &[&str],
) -> TokenStream {
    let variants = constants
        .into_iter()
        .filter(|constant| constant.notation() != Some(BACKWARDS_COMPATIBLE_ALIAS_COMMENT))
        .map(|constant| {
            let variant_ident = constant.variant_ident(enum_name, vendors);
            let notation = constant.doc_attribute();
            let constant = constant.constant(enum_name, vendors);
            let value = if let Constant::Alias(_) = &constant {
                quote!(#constant)
            } else {
                quote!(Self(#constant))
            };

            quote! {
                #notation
                pub const #variant_ident: Self = #value;
            }
        });

    quote! {
        impl #ident {
            #(#variants)*
        }
    }
}

pub fn generate_enum<'s, 'a: 's>(
    enum_: &'s Enums<'a>,
    const_cache: &mut HashSet<&'s str>,
    const_values: &mut BTreeMap<Ident, ConstantTypeInfo>,
    bitflags_cache: &mut HashSet<Ident>,
    vendors: &[&str],
) -> EnumType {
    let name = enum_.name.as_ref();
    let ident = name_to_tokens(name);

    let mut values = Vec::new();
    let khronos_link = khronos_link(name);

    let enum_type = match &enum_.values {
        EnumsValues::Constants(_) => unreachable!(),
        EnumsValues::Enum(children, _) => {
            let (struct_attribute, special_quote) = match name {
                //"StructureType" => generate_structure_type(&_name, _enum, create_info_constants),
                "VkResult" => (
                    Some(quote!(#[must_use])),
                    Some(generate_result(&ident, enum_, vendors)),
                ),
                _ => (None, None),
            };

            for v in children.values() {
                if v.notation() == Some(BACKWARDS_COMPATIBLE_ALIAS_COMMENT) {
                    continue;
                }
                const_cache.insert(v.name());
                values.push(ConstantMatchInfo {
                    ident: v.variant_ident(name, vendors),
                    is_alias: v.is_alias(),
                });
            }

            let impl_block = bitflags_impl_block(&ident, name, children.values(), vendors);
            let enum_quote = quote! {
                #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
                #[repr(transparent)]
                #[doc = #khronos_link]
                #struct_attribute
                pub struct #ident(pub(crate) i32);
                impl #ident {
                    #[inline]
                    pub const fn from_raw(x: i32) -> Self { Self(x) }
                    #[inline]
                    pub const fn as_raw(self) -> i32 { self.0 }
                }
                #impl_block
            };
            let q = quote! {
                #enum_quote
                #special_quote

            };
            EnumType::Enum(q)
        }
        EnumsValues::Bitmask(children) => {
            let type_ = if enum_.bit_width.map(|b| b.get() as _) == Some(64u32) {
                quote!(Flags64)
            } else {
                quote!(Flags)
            };

            for v in children.values() {
                if v.notation() == Some(BACKWARDS_COMPATIBLE_ALIAS_COMMENT) {
                    continue;
                }
                const_cache.insert(v.name());
                values.push(ConstantMatchInfo {
                    ident: v.variant_ident(name, vendors),
                    is_alias: v.is_alias(),
                });
            }

            if !bitflags_cache.insert(ident.clone()) {
                EnumType::Bitflags(quote! {})
            } else {
                let impl_bitflags = bitflags_impl_block(&ident, name, children.values(), vendors);
                let q = quote! {
                    #[repr(transparent)]
                    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
                    #[doc = #khronos_link]
                    pub struct #ident(pub(crate) #type_);
                    vk_bitflags_wrapped!(#ident, #type_);
                    #impl_bitflags
                };
                EnumType::Bitflags(q)
            }
        }
    };
    const_values.insert(
        ident,
        ConstantTypeInfo {
            values,
            bitwidth: enum_.bit_width.map(|b| b.get() as _),
        },
    );
    enum_type
}

pub fn generate_result(ident: &Ident, enum_: &Enums, vendors: &[&str]) -> TokenStream {
    let notation = if let EnumsValues::Enum(children, _unused) = &enum_.values {
        children
            .values()
            .filter_map(get_variant!(DefinitionOrAlias::Definition))
            .map(
                |ValueEnum {
                     name: variant_name,
                     comment,
                     ..
                 }| {
                    let variant_ident = variant_ident(enum_.name.as_ref(), variant_name, vendors);
                    let notation = comment.as_deref().unwrap_or_default();
                    quote! {
                        Self::#variant_ident => Some(#notation)
                    }
                },
            )
    } else {
        unreachable!()
    };

    quote! {
        impl ::std::error::Error for #ident {}
        impl fmt::Display for #ident {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                let name = match *self {
                    #(#notation),*,
                    _ => None,
                };
                if let Some(x) = name {
                    fmt.write_str(x)
                } else {
                    // If we don't have a nice message to show, call the generated `Debug` impl
                    // which includes *all* enum variants, including those from extensions.
                    <Self as fmt::Debug>::fmt(self, fmt)
                }
            }
        }
    }
}

fn is_static_array(field: &Member) -> bool {
    field.array_shape().is_some()
}
pub fn derive_default(name: &str, members: &[&Member], has_lifetime: bool) -> Option<TokenStream> {
    let name = name_to_tokens(name);
    let is_structure_type = |field: &Member| field.type_name.as_identifier() == "VkStructureType";

    // FIXME do we need to hardcode this, or could we change these to Option<NonNull<c_void>> then they would implment default
    // These are also pointers, and therefor also don't implement Default. The spec
    // also doesn't mark them as pointers
    let handles = [
        "LPCWSTR",
        "HANDLE",
        "HINSTANCE",
        "HWND",
        "HMONITOR",
        "IOSurfaceRef",
        // the following are actualy specified as pointers, just behind currently un-parsed macros
        "MTLBuffer_id",
        "MTLCommandQueue_id",
        "MTLDevice_id",
        "MTLSharedEvent_id",
        "MTLTexture_id",
    ];
    let contains_ptr = members.iter().any(|field| field.pointer_kind.is_some());
    let contains_structure_type = members.iter().copied().any(is_structure_type);
    let contains_static_array = members.iter().copied().any(is_static_array);
    if !(contains_ptr || contains_structure_type || contains_static_array) {
        return None;
    };
    let default_fields = members.iter().map(|field| {
        let param_ident = field.param_ident();
        if is_structure_type(field) {
            if field.values.is_some() {
                quote! {
                    #param_ident: Self::STRUCTURE_TYPE
                }
            } else {
                quote! {
                    #param_ident: unsafe { ::std::mem::zeroed() }
                }
            }
        } else if field.pointer_kind.is_some() {
            if field.is_const {
                quote!(#param_ident: ::std::ptr::null())
            } else {
                quote!(#param_ident: ::std::ptr::null_mut())
            }
        } else if is_static_array(field) || handles.contains(&field.type_name.as_identifier()) {
            quote! {
                #param_ident: unsafe { ::std::mem::zeroed() }
            }
        } else {
            let ty = field.type_tokens(false);
            quote! {
                #param_ident: #ty::default()
            }
        }
    });
    let lifetime = has_lifetime.then(|| quote!(<'_>));
    let marker = has_lifetime.then(|| quote!(_marker: PhantomData,));
    let q = quote! {
        impl ::std::default::Default for #name #lifetime {
            #[inline]
            fn default() -> Self {
                Self {
                    #(
                        #default_fields,
                    )*
                    #marker
                }
            }
        }
    };
    Some(q)
}
pub fn derive_debug(
    name: &str,
    members: &[&Member],
    union_types: &HashSet<&str>,
    has_lifetime: bool,
) -> Option<TokenStream> {
    let name = name_to_tokens(name);
    let contains_pfn = members.iter().any(|field| field.name.contains("pfn"));
    let contains_static_array = members
        .iter()
        .any(|x| is_static_array(x) && x.type_name == TypeSpecifier::Char);
    let contains_union = members
        .iter()
        .any(|field| union_types.contains(field.type_name.as_identifier()));
    let is_bitfield = members.iter().any(|field| field.bitfield_size().is_some());
    if !(contains_union || contains_static_array || contains_pfn || is_bitfield) {
        return None;
    }
    let debug_fields = members.iter().map(|field| {
        let param_ident = field.param_ident();
        let param_str = param_ident.to_string();
        let debug_value = if is_static_array(field) && field.type_name == TypeSpecifier::Char {
            quote! {
                &unsafe {
                    ::std::ffi::CStr::from_ptr(self.#param_ident.as_ptr())
                }
            }
        } else if param_str.contains("pfn") {
            quote! {
                &(self.#param_ident.map(|x| x as *const ()))
            }
        } else if union_types.contains(field.type_name.as_identifier()) {
            quote!(&"union")
        } else if is_bitfield {
            let getter = format_ident!("get_{}", param_str);
            quote!(&self.#getter())
        } else {
            quote! {
                &self.#param_ident
            }
        };
        quote! {
            .field(#param_str, #debug_value)
        }
    });
    let name_str = name.to_string();
    let lifetime = has_lifetime.then(|| quote!(<'_>));
    let q = quote! {
        #[cfg(feature = "debug")]
        impl fmt::Debug for #name #lifetime {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_struct(#name_str)
                #(#debug_fields)*
                .finish()
            }
        }
    };
    Some(q)
}

pub fn derive_setters(
    name_: &str,
    members: &[&Member],
    structextends: &[&str],
    root_structs: &HashSet<Ident>,
    has_lifetimes: &HashSet<Ident>,
    vendors: &[&str],
) -> Option<TokenStream> {
    if name_ == "VkBaseInStructure"
        || name_ == "VkBaseOutStructure"
        || name_ == "VkAccelerationStructureInstanceKHR"
    {
        return None;
    }

    let name = name_to_tokens(name_);

    let next_field = members.iter().find(|field| field.param_ident() == "p_next");

    let structure_type_field = members.iter().find(|field| field.param_ident() == "s_type");

    // Must either have both, or none:
    assert_eq!(next_field.is_some(), structure_type_field.is_some());

    let is_bitfield = members.iter().any(|field| field.bitfield_size().is_some());

    let nofilter_count_members = [
        ("VkPipelineViewportStateCreateInfo", "pViewports"),
        ("VkPipelineViewportStateCreateInfo", "pScissors"),
        ("VkDescriptorSetLayoutBinding", "pImmutableSamplers"),
    ];
    let filter_members: Vec<String> = members
        .iter()
        .filter_map(|field| {
            let field_name = field.name.as_ref();

            // Associated _count members
            if field.dynamic_shape.is_some() {
                if let Some(
                    DynamicShapeKind::Single(DynamicLength::Parameterized(array_size))
                    | DynamicShapeKind::Double(DynamicLength::Parameterized(array_size), _),
                ) = &field.dynamic_shape
                {
                    if !nofilter_count_members.contains(&(name_, field_name)) {
                        return Some(array_size.to_string());
                    }
                }
            }

            // VkShaderModuleCreateInfo requires a custom setter
            if field_name == "codeSize" {
                return Some(field_name.to_string());
            }

            None
        })
        .collect();

    let setters = members.iter().filter_map(|field| {
        if is_bitfield {
            return None;
        }
        let param_ident = field.param_ident();
        let param_ty_tokens = field.safe_type_tokens(quote!('a));

        let param_ident_string = param_ident.to_string();
        if param_ident_string == "s_type" || param_ident_string == "p_next" {
            return None;
        }

        let param_ident_short = param_ident_string
            .strip_prefix("p_")
            .or_else(|| param_ident_string.strip_prefix("pp_"))
            .unwrap_or(&param_ident_string);
        let mut param_ident_short = format_ident!("{}", param_ident_short);

        {
            let name = field.name;
            // Filter
            if filter_members.iter().any(|n| *n == *name) {
                return None;
            }

            // Unique cases
            if name == "pCode" {
                return Some(quote!{
                    #[inline]
                    pub fn code(mut self, code: &'a [u32]) -> Self {
                        self.code_size = code.len() * 4;
                        self.p_code = code.as_ptr();
                        self
                    }
                });
            }

            if name == "pSampleMask" {
                return Some(quote!{
                    /// Sets `p_sample_mask` to `null` if the slice is empty. The mask will
                    /// be treated as if it has all bits set to `1`.
                    ///
                    /// See <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkPipelineMultisampleStateCreateInfo.html#_description>
                    /// for more details.
                    #[inline]
                    pub fn sample_mask(mut self, sample_mask: &'a [SampleMask]) -> Self {
                        self.p_sample_mask = if sample_mask.is_empty() {
                            std::ptr::null()
                        } else {
                            sample_mask.as_ptr()
                        };
                        self
                    }
                });
            }
        }

        // TODO: Improve in future when https://github.com/rust-lang/rust/issues/53667 is merged id:6
        if let Some(kind) = &field.pointer_kind {
            if field.type_name == TypeSpecifier::Char && matches!(kind, PointerKind::Single {..}) {
                assert!(matches!(field.dynamic_shape, Some(DynamicShapeKind::Single(DynamicLength::NullTerminated))));
                return Some(quote!{
                    #[inline]
                    pub fn #param_ident_short(mut self, #param_ident_short: &'a ::std::ffi::CStr) -> Self {
                        self.#param_ident = #param_ident_short.as_ptr();
                        self
                    }
                });
            }

            let is_const = field.is_const;

            if let Some(dyn_shape) = &field.dynamic_shape {
                let is_array_ish = matches!(dyn_shape, DynamicShapeKind::Expression { .. } | DynamicShapeKind::Single(DynamicLength::Parameterized(_)) | DynamicShapeKind::Double(DynamicLength::Parameterized(_), _));
                if is_array_ish {
                    let mut slice_param_ty_tokens = field.safe_type_tokens(quote!('a));

                    let mut ptr = if is_const {
                        quote!(.as_ptr())
                    } else {
                        quote!(.as_mut_ptr())
                    };

                    // Interpret void array as byte array
                    if field.type_name == TypeSpecifier::Void && matches!(field.pointer_kind, Some(PointerKind::Single {..})) {
                        slice_param_ty_tokens = quote!([u8]);
                        ptr = quote!(#ptr.cast());
                    };

                    let set_size_stmt = if let DynamicShapeKind::Expression { c_expr, .. } = &dyn_shape {
                        // this is a pointer to a piece of memory with statically known size.
                        let c_size = expression_tokens(c_expr, &BTreeMap::new(), false);
                        let inner_type = field.inner_type_tokens(None);

                        slice_param_ty_tokens = quote!([#inner_type; #c_size]);
                        ptr = quote!();
                        None
                    } else if let DynamicShapeKind::Single(DynamicLength::Parameterized(p)) | DynamicShapeKind::Double(DynamicLength::Parameterized(p), _) = &dyn_shape {
                        // Deal with a "special" 2D dynamic array with an inner size of 1 (effectively an array containing pointers to single objects)
                        if let DynamicShapeKind::Double(DynamicLength::Parameterized(_), DynamicLength::Static(_)) = &dyn_shape {
                            param_ident_short = format_ident!("{}_ptrs", param_ident_short);
                            slice_param_ty_tokens = field.safe_type_tokens(quote!('a));
                            ptr = quote!(#ptr.cast());
                        }
                        let array_size_ident = format_ident!("{}", p.to_snake_case().as_str());

                        let size_field = members.iter().find(|m| m.name == *p).unwrap();

                        let cast = if size_field.type_name.as_identifier() == "size_t" {
                            None
                        }else{
                            Some(quote!(as _))
                        };

                        Some(quote!(self.#array_size_ident = #param_ident_short.len()#cast;))
                    } else {
                        unreachable!()
                    };

                    let mutable = (!is_const).then(|| quote!(mut));

                    return Some(quote! {
                        #[inline]
                        pub fn #param_ident_short(mut self, #param_ident_short: &'a #mutable #slice_param_ty_tokens) -> Self {
                            #set_size_stmt
                            self.#param_ident = #param_ident_short #ptr;
                            self
                        }
                    });
                }
            }
        }

        if field.type_name.as_identifier() == "VkBool32" {
            return Some(quote!{
                #[inline]
                pub fn #param_ident_short(mut self, #param_ident_short: bool) -> Self {
                    self.#param_ident = #param_ident_short.into();
                    self
                }
            });
        }

        let param_ty_tokens = if is_opaque_type(field.type_name.as_identifier()) {
            //  Use raw pointers for void/opaque types
            field.type_tokens(false)
        } else {
            param_ty_tokens
        };

        let lifetime = has_lifetimes
            .contains(&name_to_tokens(field.type_name.as_identifier()))
            .then(|| quote!(<'a>));

        Some(quote!{
            #[inline]
            pub fn #param_ident_short(mut self, #param_ident_short: #param_ty_tokens #lifetime) -> Self {
                self.#param_ident = #param_ident_short;
                self
            }
        })
    });

    let bitfield_fns = if is_bitfield {
        Some(members.iter().scan(0u8, |offset, field| {
            let param_ident = field.param_ident();
            let param_ident_string = param_ident.to_string();
            let getter_ident = format_ident!("get_{}", param_ident_string);

            let size = field.bitfield_size().unwrap();

            let bit_len = size.get();
            let val_ty = if bit_len <= 1 {
                quote!(bool)
            } else if bit_len <= 8 {
                quote!(u8)
            } else if bit_len <= 16 {
                quote!(u16)
            } else if bit_len <= 32 {
                quote!(u32)
            } else if bit_len <= 64 {
                quote!(u64)
            } else if bit_len <= 128 {
                quote!(u128)
            } else {
                unreachable!()
            };

            let shift = *offset;
            let param_ty = name_to_tokens(field.type_name.as_identifier());
            let into_val_ty = if bit_len == 1 {
                quote!(!=0)
            } else {
                quote!(as #val_ty)
            };

            *offset += bit_len;
            Some(quote! {
                #[inline]
                pub fn #param_ident(mut self, #param_ident: #val_ty) -> Self {
                    const SHIFT: u8 = #shift;
                    const BIT_LEN: u8 = #bit_len;
                    const MASK: #param_ty = (1 << BIT_LEN) - 1;
                    const CLEAR_MASK: #param_ty = !(MASK << SHIFT);
                    self.bytes &= CLEAR_MASK;
                    self.bytes |= ((#param_ident as #param_ty) & MASK) << SHIFT;
                    self
                }

                #[inline]
                pub fn #getter_ident(&self) -> #val_ty {
                    const SHIFT: u8 = #shift;
                    const BIT_LEN: u8 = #bit_len;
                    const MASK: #param_ty = (1 << BIT_LEN) - 1;
                    ((self.bytes >> SHIFT) & MASK) #into_val_ty
                }
            })
        }))
    } else {
        None
    };
    let bitfield_fns = bitfield_fns.into_iter().flatten();

    let extends_name = format_ident!("Extends{}", name);

    // The `p_next` field should only be considered if this struct is also a root struct
    let root_struct_next_field = next_field.filter(|_| root_structs.contains(&name));

    // We only implement a next methods for root structs with a `pnext` field.
    let next_function = root_struct_next_field.map(|next_field| {
        assert_eq!(next_field.type_name, TypeSpecifier::Void);

        let is_const = next_field.is_const;

        let mutability = if is_const { quote!(const) } else { quote!(mut) };
        quote! {
            /// Prepends the given extension struct between the root and the first pointer. This
            /// method only exists on structs that can be passed to a function directly. Only
            /// valid extension structs can be pushed into the chain.
            /// If the chain looks like `A -> B -> C`, and you call `x.push_next(&mut D)`, then the
            /// chain will look like `A -> D -> B -> C`.
            pub fn push_next<T: #extends_name>(mut self, next: &'a mut T) -> Self {
                unsafe {
                    let next_ptr = <*#mutability T>::cast(next);
                    // `next` here can contain a pointer chain. This means that we must correctly
                    // attach he head to the root and the tail to the rest of the chain
                    // For example:
                    //
                    // next = A -> B
                    // Before: `Root -> C -> D -> E`
                    // After: `Root -> A -> B -> C -> D -> E`
                    //                 ^^^^^^
                    //                 next chain
                    let last_next = ptr_chain_iter(next).last().unwrap();
                    (*last_next).p_next = self.p_next as _;
                    self.p_next = next_ptr;
                }
                self
            }
        }
    });

    // Root structs come with their own trait that structs that extend
    // this struct will implement
    let next_trait = if root_struct_next_field.is_some() {
        quote!(pub unsafe trait #extends_name {})
    } else {
        quote!()
    };

    let lifetime = has_lifetimes.contains(&name).then(|| quote!(<'a>));

    // If the struct extends something we need to implement the traits.
    let impl_extend_trait = structextends
        .iter()
        .map(|extends| format_ident!("Extends{}", name_to_tokens(extends)))
        .map(|extends| {
            // Extension structs always have a pNext, and therefore always have a lifetime.
            quote!(unsafe impl #extends for #name<'_> {})
        });

    let impl_structure_type_trait = structure_type_field.map(|s_type| {
        let value = s_type
            .values
            .as_deref()
            .expect("s_type field must have a value in `vk.xml`");

        assert!(!value.contains(','));

        let value = variant_ident("VkStructureType", value, vendors);
        quote! {
            unsafe impl #lifetime TaggedStructure for #name #lifetime {
                const STRUCTURE_TYPE: StructureType = StructureType::#value;
            }
        }
    });

    let q = quote! {
        #impl_structure_type_trait
        #(#impl_extend_trait)*
        #next_trait

        impl #lifetime #name #lifetime {
            #(#setters)*
            #(#bitfield_fns)*

            #next_function
        }
    };

    Some(q)
}

/// FIXME: At the moment `Ash` doesn't properly derive all the necessary drives
/// like Eq, Hash etc.
/// To Address some cases, you can add the name of the struct that you
/// require and add the missing derives yourself.
pub fn manual_derives(name: &str) -> TokenStream {
    match name {
        "VkClearRect" | "VkExtent2D" | "VkExtent3D" | "VkOffset2D" | "VkOffset3D" | "VkRect2D"
        | "VkSurfaceFormatKHR" => quote! {PartialEq, Eq, Hash,},
        _ => quote! {},
    }
}
pub fn generate_struct(
    name_: &str,
    members: &[&Member],
    structextends: &[&str],
    root_structs: &HashSet<Ident>,
    union_types: &HashSet<&str>,
    has_lifetimes: &HashSet<Ident>,
    vendors: &[&str],
) -> TokenStream {
    let name = name_to_tokens(name_);

    // FIXME: The following special cases only still exist because `acceleration_structure_reference`
    // has the wrong type in the reference xml (uint64_t instead of union {VkDeviceAddress, AccelerationStructureKHR})
    if name_ == "VkAccelerationStructureInstanceKHR" {
        return quote! {
            #[repr(C)]
            #[derive(Copy, Clone)]
            pub union AccelerationStructureReferenceKHR {
                pub device_handle: DeviceAddress,
                pub host_handle: AccelerationStructureKHR,
            }
            #[repr(C)]
            #[derive(Copy, Clone)]
            #[doc = "<https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureInstanceKHR.html>"]
            pub struct AccelerationStructureInstanceKHR {
                pub transform: TransformMatrixKHR,
                /// Use [`Packed24_8::new(instance_custom_index, mask)`][Packed24_8::new()] to construct this field
                pub instance_custom_index_and_mask: Packed24_8,
                /// Use [`Packed24_8::new(instance_shader_binding_table_record_offset, flags)`][Packed24_8::new()] to construct this field
                pub instance_shader_binding_table_record_offset_and_flags: Packed24_8,
                pub acceleration_structure_reference: AccelerationStructureReferenceKHR,
            }
        };
    }

    if name_ == "VkAccelerationStructureSRTMotionInstanceNV" {
        return quote! {
            #[repr(C)]
            #[derive(Copy, Clone)]
            #[doc = "<https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureSRTMotionInstanceNV.html>"]
            pub struct AccelerationStructureSRTMotionInstanceNV {
                pub transform_t0: SRTDataNV,
                pub transform_t1: SRTDataNV,
                /// Use [`Packed24_8::new(instance_custom_index, mask)`][Packed24_8::new()] to construct this field
                pub instance_custom_index_and_mask: Packed24_8,
                /// Use [`Packed24_8::new(instance_shader_binding_table_record_offset, flags)`][Packed24_8::new()] to construct this field
                pub instance_shader_binding_table_record_offset_and_flags: Packed24_8,
                pub acceleration_structure_reference: AccelerationStructureReferenceKHR,
            }
        };
    }

    if name_ == "VkAccelerationStructureMatrixMotionInstanceNV" {
        return quote! {
            #[repr(C)]
            #[derive(Copy, Clone)]
            #[doc = "<https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/AccelerationStructureMatrixMotionInstanceNV.html>"]
            pub struct AccelerationStructureMatrixMotionInstanceNV {
                pub transform_t0: TransformMatrixKHR,
                pub transform_t1: TransformMatrixKHR,
                /// Use [`Packed24_8::new(instance_custom_index, mask)`][Packed24_8::new()] to construct this field
                pub instance_custom_index_and_mask: Packed24_8,
                /// Use [`Packed24_8::new(instance_shader_binding_table_record_offset, flags)`][Packed24_8::new()] to construct this field
                pub instance_shader_binding_table_record_offset_and_flags: Packed24_8,
                pub acceleration_structure_reference: AccelerationStructureReferenceKHR,
            }
        };
    }

    let is_bitfield = members.iter().any(|field| field.bitfield_size().is_some());
    let bitfield_ty = if is_bitfield {
        assert!(
            members
                .iter()
                .all(|field| { field.bitfield_size().is_some() }),
            "If any member is a bitfield, than all most be one"
        );
        assert!(
            members
                .windows(2)
                .all(|fs| fs[0].type_name == fs[1].type_name),
            "All bitfields must have the same primitive they fit in"
        );
        Some(name_to_tokens(members[0].type_name.as_identifier()))
    } else {
        None
    };

    let params = members
        .iter()
        .filter_map(|field| {
            let param_ident = field.param_ident();
            let param_ty_tokens = if field.type_name.as_identifier() == name_ {
                FieldLike {
                    type_name: TypeSpecifier::TypedefName("Self".into()),
                    ..field.base.clone()
                }
                .type_tokens(false)
            } else if field.bitfield_size().is_some() {
                return None;
            } else {
                let lifetime = has_lifetimes
                    .contains(&name_to_tokens(field.type_name.as_identifier()))
                    .then(|| quote!(<'a>));
                let ty = field.type_tokens(false);
                quote!(#ty #lifetime)
            };
            Some(quote! {pub #param_ident: #param_ty_tokens})
        })
        .chain(bitfield_ty.map(|ty| quote! {pub bytes: #ty}));

    let has_lifetime = has_lifetimes.contains(&name);
    let (lifetimes, marker) = has_lifetime
        .then(|| (quote!(<'a>), quote!(pub _marker: PhantomData<&'a ()>,)))
        .unzip();

    let debug_tokens = derive_debug(name_, members, union_types, has_lifetime);
    let default_tokens = derive_default(name_, members, has_lifetime);
    let setter_tokens = derive_setters(
        name_,
        members,
        structextends,
        root_structs,
        has_lifetimes,
        vendors,
    );
    let manual_derive_tokens = manual_derives(name_);
    let dbg_str = debug_tokens
        .is_none()
        .then(|| quote!(#[cfg_attr(feature = "debug", derive(Debug))]));
    let default_str = default_tokens.is_none().then(|| quote!(Default,));
    let khronos_link = khronos_link(&name_);
    quote! {
        #[repr(C)]
        #dbg_str
        #[derive(Copy, Clone, #default_str #manual_derive_tokens)]
        #[doc = #khronos_link]
        pub struct #name #lifetimes {
            #(#params,)*
            #marker
        }
        #debug_tokens
        #default_tokens
        #setter_tokens
    }
}

pub fn generate_handle(handle: &HandleType) -> Option<TokenStream> {
    let HandleType {
        name,
        handle_kind,
        obj_type_enum: _,
        parent: _,
        ..
    } = handle;
    let khronos_link = khronos_link(name);
    let name = name.strip_prefix("Vk").unwrap_or(name);
    let ty = format_ident!("{}", name.to_shouty_snake_case());
    let name = format_ident!("{}", name);

    match handle_kind {
        HandleKind::Dispatch => Some(quote! {
            define_handle!(#name, #ty, doc = #khronos_link);
        }),
        HandleKind::NoDispatch => Some(quote! {
            handle_nondispatchable!(#name, #ty, doc = #khronos_link);
        }),
    }
}
fn generate_funcptr(fnptr: &FnPtrType) -> TokenStream {
    let name = format_ident!("{}", fnptr.name);
    let ret_ty_tokens = if fnptr.return_type_name == TypeSpecifier::Void
        && fnptr.return_type_pointer_kind.is_none()
    {
        quote!()
    } else {
        let ty = name_to_tokens(fnptr.return_type_name.as_identifier());
        let ptr = WrappedPointerKind {
            pointer_kind: fnptr.return_type_pointer_kind.as_ref(),
            is_const: false,
        }
        .to_tokens();
        quote!(-> #ptr #ty)
    };
    let params = fnptr.params.iter().flatten().map(|field| {
        let ident = field.param_ident();
        let type_tokens = field.type_tokens(true);
        quote! {
            #ident: #type_tokens
        }
    });
    let khronos_link = khronos_link(&fnptr.name);
    quote! {
        #[allow(non_camel_case_types)]
        #[doc = #khronos_link]
        pub type #name = Option<unsafe extern "system" fn(#(#params),*) #ret_ty_tokens>;
    }
}

fn generate_union(name_: &str, members: &[&Member], has_lifetimes: &HashSet<Ident>) -> TokenStream {
    let name = name_to_tokens(name_);
    let fields = members.iter().map(|field| {
        let name = field.param_ident();
        let ty = field.type_tokens(false);
        let lifetime = has_lifetimes
            .contains(&name_to_tokens(field.type_name.as_identifier()))
            .then(|| quote!(<'a>));
        quote! {
            pub #name: #ty #lifetime
        }
    });
    let khronos_link = khronos_link(&name_);
    let lifetime = has_lifetimes.contains(&name).then(|| quote!(<'a>));
    quote! {
        #[repr(C)]
        #[derive(Copy, Clone)]
        #[doc = #khronos_link]
        pub union #name #lifetime {
            #(#fields),*
        }
        impl #lifetime ::std::default::Default for #name #lifetime {
            #[inline]
            fn default() -> Self {
                unsafe { ::std::mem::zeroed() }
            }
        }
    }
}
/// Root structs are all structs that are extended by other structs.
pub fn root_structs(definitions: &[&Type]) -> HashSet<Ident> {
    let mut root_structs = HashSet::new();
    // Loop over all structs and collect their extends
    for definition in definitions {
        if let Type::Struct(DefinitionOrAlias::Definition(StructType { struct_extends, .. })) =
            definition
        {
            root_structs.extend(struct_extends.iter().flatten().map(|e| name_to_tokens(e)));
        };
    }
    root_structs
}
pub fn generate_definition<'s>(
    definition: &'s Type,
    union_types: &HashSet<&str>,
    root_structs: &HashSet<Ident>,
    has_lifetimes: &HashSet<Ident>,
    bitflags_cache: &mut HashSet<Ident>,
    const_values: &mut BTreeMap<Ident, ConstantTypeInfo>,
    identifier_renames: &mut BTreeMap<&'s str, Ident>,
    vendors: &[&str],
) -> Option<TokenStream> {
    match definition {
        Type::Define(DefineType::Macro(define)) => {
            Some(generate_define(define, identifier_renames))
        }
        // Ignore forward declarations
        Type::BaseType(BaseTypeType::TypeDef(typedef)) => Some(generate_typedef(typedef)),
        Type::Bitmask(DefinitionOrAlias::Definition(BitmaskType {
            name,
            is_64bits,
            has_bitvalues: false,
        })) => generate_bitmask(name, *is_64bits, bitflags_cache, const_values),
        Type::Handle(DefinitionOrAlias::Definition(handle)) => generate_handle(handle),
        Type::FnPtr(fp) => Some(generate_funcptr(fp)),
        Type::Struct(DefinitionOrAlias::Definition(StructType {
            name,
            members,
            struct_extends,
            ..
        })) => Some(generate_struct(
            name,
            (members.values().collect::<Vec<_>>()).as_slice(),
            struct_extends.as_deref().unwrap_or_default(),
            root_structs,
            union_types,
            has_lifetimes,
            vendors,
        )),
        Type::Union(UnionType { name, members, .. }) => Some(generate_union(
            name,
            (members.values().collect::<Vec<_>>()).as_slice(),
            has_lifetimes,
        )),
        _ => None,
    }
}
pub fn generate_feature<'s, 'a: 's>(
    feature: &'s Feature<'a>,
    commands: &CommandMap<'a, 's>,
    fn_cache: &mut HashSet<&'s str>,
) -> TokenStream {
    let (static_commands, entry_commands, device_commands, instance_commands) = feature
        .requires
        .iter()
        .flat_map(|r| r.values.values())
        .filter_map(get_variant!(RequireValue::Command { name }))
        .filter_map(|cmd_ref| commands.get(cmd_ref))
        .fold(
            (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            |mut accs, &cmd_ref| {
                let acc = match cmd_ref.function_type() {
                    FunctionType::Static => &mut accs.0,
                    FunctionType::Entry => &mut accs.1,
                    FunctionType::Device => &mut accs.2,
                    FunctionType::Instance => &mut accs.3,
                };
                acc.push(cmd_ref);
                accs
            },
        );
    let version = feature.version_string();
    let static_fn = if feature.is_version(1, 0) {
        generate_function_pointers(
            &format_ident!("StaticFn"),
            &static_commands,
            &HashMap::new(),
            fn_cache,
        )
    } else {
        quote! {}
    };
    let entry = generate_function_pointers(
        &format_ident!("EntryFnV{}", version),
        &entry_commands,
        &HashMap::new(),
        fn_cache,
    );
    let instance = generate_function_pointers(
        &format_ident!("InstanceFnV{}", version),
        &instance_commands,
        &HashMap::new(),
        fn_cache,
    );
    let device = generate_function_pointers(
        &format_ident!("DeviceFnV{}", version),
        &device_commands,
        &HashMap::new(),
        fn_cache,
    );
    quote! {
        #static_fn
        #entry
        #instance
        #device
    }
}

pub fn constant_name(name: &str) -> &str {
    name.strip_prefix("VK_").unwrap_or(name)
}

pub fn generate_constant<'s>(
    constant: &'s ConstantEnum,
    cache: &mut HashSet<&'s str>,
    vendors: &[&str],
) -> Option<TokenStream> {
    cache.insert(constant.name.as_ref());
    let name = constant_name(&constant.name);
    let ident = format_ident!("{}", name);
    let notation = constant.doc_attribute();

    let ty = if name == "TRUE" || name == "FALSE" {
        format_ident!("Bool32")
    // FIXME: Rust only accepts `usize` values for array sizes, so either change the type of the constants or cast at array defintion
    } else if (constant.name.contains("SIZE") || constant.name.contains("MAX"))
        && constant.type_name == "uint32_t"
    {
        format_ident!("usize")
    } else {
        name_to_tokens(constant.type_name.as_ref())
    };
    let c = constant.constant("", vendors);
    Some(quote! {
        #notation
        pub const #ident: #ty = #c;
    })
}

// FIXME: Remove when `video.xml` is fixed
pub fn generate_video_constant<'s>(
    constant: &'s RequireEnum,
    cache: &mut HashSet<&'s str>,
) -> Option<TokenStream> {
    let name = constant.name.as_deref().unwrap();
    cache.insert(name);
    let name = constant_name(name);
    let ident = format_ident!("{}", name);
    let notation = constant.doc_attribute();

    let ty = if name == "TRUE" || name == "FALSE" {
        format_ident!("Bool32")
    } else {
        // name_to_tokens(constant.type_name.as_ref())
        format_ident!("usize")
    };
    let c = constant.constant("", &[]);
    Some(quote! {
        #notation
        pub const #ident: #ty = #c;
    })
}

pub fn generate_feature_extension<'s, 'a: 's>(
    feature: &'a Feature,
    const_cache: &mut HashSet<&'s str>,
    const_values: &mut BTreeMap<Ident, ConstantTypeInfo>,
    vendors: &[&str],
) -> TokenStream {
    generate_extension_constants(
        &feature.name,
        0,
        &feature.requires,
        const_cache,
        const_values,
        vendors,
    )
}

pub struct ConstantMatchInfo {
    pub ident: Ident,
    pub is_alias: bool,
}

#[derive(Default)]
pub struct ConstantTypeInfo {
    values: Vec<ConstantMatchInfo>,
    bitwidth: Option<u32>,
}

pub struct ConstDebugs {
    core: TokenStream,
    extras: TokenStream,
}

pub fn generate_const_debugs(const_values: &BTreeMap<Ident, ConstantTypeInfo>) -> ConstDebugs {
    let mut core = Vec::new();
    let mut extras = Vec::new();
    for (ty, values) in const_values {
        let ConstantTypeInfo { values, bitwidth } = values;
        let out = if ty.to_string().contains("Flags") {
            let cases = values.iter().filter_map(|value| {
                if value.is_alias {
                    None
                } else {
                    let ident = &value.ident;
                    let name = ident.to_string();
                    Some(quote! { (#ty::#ident.0, #name) })
                }
            });

            let type_ = if bitwidth == &Some(64u32) {
                quote!(Flags64)
            } else {
                quote!(Flags)
            };

            quote! {
                impl fmt::Debug for #ty {
                    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                        const KNOWN: &[(#type_, &str)] = &[#(#cases),*];
                        debug_flags(f, KNOWN, self.0)
                    }
                }
            }
        } else {
            let cases = values.iter().filter_map(|value| {
                if value.is_alias {
                    None
                } else {
                    let ident = &value.ident;
                    let name = ident.to_string();
                    Some(quote! { Self::#ident => Some(#name), })
                }
            });
            quote! {
                impl fmt::Debug for #ty {
                    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                        let name = match *self {
                            #(#cases)*
                            _ => None,
                        };
                        if let Some(x) = name {
                            f.write_str(x)
                        } else {
                            self.0.fmt(f)
                        }
                    }
                }
            }
        };
        if ty == "Result" || ty == "ObjectType" {
            core.push(out);
        } else {
            extras.push(out);
        }
    }

    ConstDebugs {
        core: quote! {
            #(#core)*
        },
        extras: quote! {
            #(#extras)*
        },
    }
}

pub fn generate_aliases_of_types<'i, 'a: 'i>(
    types: impl IntoIterator<Item = &'i Type<'a>>,
    has_lifetimes: &HashSet<Ident>,
    ty_cache: &mut HashSet<Ident>,
) -> TokenStream {
    let aliases = types.into_iter().filter_map(|ty| {
        let (Type::Bitmask(DefinitionOrAlias::Alias(alias)) | Type::Handle(DefinitionOrAlias::Alias(alias)) | Type::Enum(DefinitionOrAlias::Alias(alias)) | Type::Struct(DefinitionOrAlias::Alias(alias))) = ty else {
            return None
        };
        let name_ident = name_to_tokens(alias.name.as_ref());
        if !ty_cache.insert(name_ident.clone()) {
            return None;
        };
        let alias_ident = name_to_tokens(alias.alias.as_ref());
        let tokens = if has_lifetimes.contains(&alias_ident) {
            quote!(pub type #name_ident<'a> = #alias_ident<'a>;)
        } else {
            quote!(pub type #name_ident = #alias_ident;)
        };
        Some(tokens)
    });
    quote! {
        #(#aliases)*
    }
}
pub fn write_source_code<P: AsRef<Path>>(vk_headers_dir: &Path, src_dir: P) {
    let vk_xml =
        fs::read_to_string(vk_headers_dir.join("registry/vk.xml")).expect("Invalid xml file");
    let video_xml =
        fs::read_to_string(vk_headers_dir.join("registry/video.xml")).expect("Invalid xml file");
    let vk_xml_doc = vulkan_parse::Document::parse(&vk_xml).expect("Invalid xml");
    let video_xml_doc = vulkan_parse::Document::parse(&video_xml).expect("Invalid xml");
    let spec =
        parse_registry(&vk_xml_doc).unwrap_or_else(|e| panic!("Failed to parse vk.xml {}", e));
    let video_spec = parse_registry(&video_xml_doc)
        .unwrap_or_else(|e| panic!("Failed to parse video.xml {}", e));

    let Registry(registry) = spec;
    let Registry(video_registry) = video_spec;

    let vendors_vec: Vec<&str> = registry
        .values()
        .filter_map(get_variant!(Items::Tags { tags }))
        .flatten()
        .map(|tag| tag.name)
        .collect();
    let vendors: &[&str] = &vendors_vec;

    let extensions: Vec<&Extension> = registry
        .values()
        .filter_map(get_variant!(Items::Extensions { extensions }))
        .flatten()
        .filter_map(get_variant!(WrappedExtension::Extension))
        .collect();

    let cmd_aliases: HashMap<&str, &str> = registry
        .values()
        .filter_map(get_variant!(Items::Commands { commands }))
        .flat_map(CommentendChildren::values)
        .filter_map(get_variant!(DefinitionOrAlias::Alias))
        .map(|Alias { name, alias, .. }| (name.as_ref(), alias.as_ref()))
        .collect();

    let commands: HashMap<&str, &Command> = registry
        .values()
        .filter_map(get_variant!(Items::Commands { commands }))
        .flat_map(CommentendChildren::values)
        .filter_map(get_variant!(DefinitionOrAlias::Definition))
        .map(|cmd| (cmd.proto.name.as_ref(), (cmd)))
        .collect();

    let features: Vec<&Feature> = registry
        .values()
        .filter_map(get_variant!(Items::Features))
        .collect();

    // FIXME sperate out aliases
    let definitions: Vec<&Type> = registry
        .values()
        .chain(video_registry.values())
        .filter_map(get_variant!(Items::Types { types }))
        .flat_map(CommentendChildren::values)
        .collect();

    // video.xml diverges from vk.xml in that it doesn't put it's hardcoded constants in a bare <enums>
    // but instead places them inside the <require> of confusing pseudo-extensions
    //
    // This is a hacky workaround, but would need to be fixed upstream in vulkan-docs itself
    let video_constants: Vec<&RequireEnum> = video_registry
        .values()
        .filter_map(get_variant!(Items::Extensions { extensions }))
        .flatten()
        .filter_map(get_variant!(WrappedExtension::PseudoExtension))
        .flat_map(|PseudoExtension { requires, .. }| requires)
        .flat_map(|Require { values, .. }| values.values())
        .filter_map(get_variant!(RequireValue::Enum))
        .filter(|e| {
            e.name.as_deref().map_or(true, |n| !n.starts_with("VK"))
                && matches!(e.value, Some(RequireValueEnum::Value(_)))
        })
        .collect();

    let constants: Vec<_> = registry
        .values()
        .filter_map(get_variant!(Items::Enums))
        .map(|e| &e.values)
        .filter_map(get_variant!(EnumsValues::Constants))
        .flat_map(CommentendChildren::values)
        .filter_map(get_variant!(DefinitionOrAlias::Definition))
        .collect();

    let mut fn_cache = HashSet::new();
    let mut bitflags_cache = HashSet::new();
    let mut const_cache = HashSet::new();

    let mut const_values: BTreeMap<Ident, ConstantTypeInfo> = BTreeMap::new();

    let (enum_code, bitflags_code) = registry
        .values()
        .chain(video_registry.values())
        .filter_map(get_variant!(Items::Enums))
        .filter(|enums| !matches!(enums.values, EnumsValues::Constants(_)))
        .map(|e| {
            generate_enum(
                e,
                &mut const_cache,
                &mut const_values,
                &mut bitflags_cache,
                vendors,
            )
        })
        .fold((Vec::new(), Vec::new()), |mut acc, elem| {
            match elem {
                EnumType::Enum(token) => acc.0.push(token),
                EnumType::Bitflags(token) => acc.1.push(token),
            };
            acc
        });

    let mut constants_code: Vec<_> = constants
        .iter()
        .filter_map(|constant| generate_constant(constant, &mut const_cache, vendors))
        .collect();
    constants_code.extend(
        video_constants
            .iter()
            .filter_map(|constant| generate_video_constant(constant, &mut const_cache)),
    );
    constants_code.push(quote! { pub const SHADER_UNUSED_NV : u32 = SHADER_UNUSED_KHR;});

    let extension_code = extensions
        .iter()
        .filter(|e| {
            // Note that there will be multiple Vulkan API variants in the future, communicated
            // through the supported= attribute:
            // https://github.com/KhronosGroup/Vulkan-Docs/issues/1549#issuecomment-855831740
            e.supported != ExtensionSupport::Disabled ||
                // VK_ANDROID_native_buffer is for internal use only, but types defined elsewhere
                // reference enum extension constants.  Exempt the extension from this check until
                // types are properly folded in with their extension (where applicable).
                e.name == "VK_ANDROID_native_buffer"
        })
        .filter_map(|ext| {
            generate_extension(
                ext,
                &commands,
                &mut const_cache,
                &mut const_values,
                &cmd_aliases,
                &mut fn_cache,
                vendors,
            )
        })
        .collect_vec();

    let union_types = definitions
        .iter()
        .filter_map(|def| match def {
            Type::Union(UnionType { name, .. }) => Some(name.as_ref()),
            _ => None,
        })
        .collect::<HashSet<&str>>();

    let mut identifier_renames = BTreeMap::new();

    // Identify structs that need a lifetime annotation
    // Note that this relies on `vk.xml` defining types before they are used,
    // as is required in C(++) too.
    let mut has_lifetimes = definitions
        .iter()
        .filter_map(|def| match def {
            Type::Struct(DefinitionOrAlias::Definition(StructType { name, members, .. })) => {
                members
                    .values()
                    .any(|x| x.pointer_kind.is_some())
                    .then(|| name_to_tokens(name))
            }
            _ => None,
        })
        .collect::<HashSet<Ident>>();
    for def in &definitions {
        match &def {
            Type::Struct(DefinitionOrAlias::Definition(StructType { name, members, .. }))
            | Type::Union(UnionType { name, members, .. }) => members
                .values()
                .any(|field| {
                    has_lifetimes.contains(&name_to_tokens(field.type_name.as_identifier()))
                })
                .then(|| has_lifetimes.insert(name_to_tokens(name))),
            _ => continue,
        };
    }

    let root_structs = root_structs(definitions.as_slice());
    let definition_code: Vec<_> = definitions
        .into_iter()
        .filter_map(|def| {
            generate_definition(
                def,
                &union_types,
                &root_structs,
                &has_lifetimes,
                &mut bitflags_cache,
                &mut const_values,
                &mut identifier_renames,
                vendors,
            )
        })
        .collect();

    let mut ty_cache = HashSet::new();
    let aliases: Vec<_> = registry
        .values()
        .filter_map(get_variant!(Items::Types { types }))
        .map(|tys| generate_aliases_of_types(tys.values(), &has_lifetimes, &mut ty_cache))
        .collect();

    let feature_code: Vec<_> = features
        .iter()
        .map(|feature| generate_feature(feature, &commands, &mut fn_cache))
        .collect();
    let feature_extensions_code: Vec<_> = features
        .iter()
        .map(|feature| {
            generate_feature_extension(feature, &mut const_cache, &mut const_values, vendors)
        })
        .collect();

    let ConstDebugs {
        core: core_debugs,
        extras: const_debugs,
    } = generate_const_debugs(&const_values);

    let src_dir = src_dir.as_ref();

    let vk_dir = src_dir.join("vk");
    std::fs::create_dir_all(&vk_dir).expect("failed to create vk dir");

    let mut vk_features_file = File::create(vk_dir.join("features.rs")).expect("vk/features.rs");
    let mut vk_definitions_file =
        File::create(vk_dir.join("definitions.rs")).expect("vk/definitions.rs");
    let mut vk_enums_file = File::create(vk_dir.join("enums.rs")).expect("vk/enums.rs");
    let mut vk_bitflags_file = File::create(vk_dir.join("bitflags.rs")).expect("vk/bitflags.rs");
    let mut vk_constants_file = File::create(vk_dir.join("constants.rs")).expect("vk/constants.rs");
    let mut vk_extensions_file =
        File::create(vk_dir.join("extensions.rs")).expect("vk/extensions.rs");
    let mut vk_feature_extensions_file =
        File::create(vk_dir.join("feature_extensions.rs")).expect("vk/feature_extensions.rs");
    let mut vk_const_debugs_file =
        File::create(vk_dir.join("const_debugs.rs")).expect("vk/const_debugs.rs");
    let mut vk_aliases_file = File::create(vk_dir.join("aliases.rs")).expect("vk/aliases.rs");

    // FIXME switch `std::os::raw::*` to `core::ffi::*` once MSRV is 1.64
    // as suggested by https://doc.rust-lang.org/std/os/raw/
    let feature_code = quote! {
        use std::os::raw::*;
        use crate::vk::bitflags::*;
        use crate::vk::definitions::*;
        use crate::vk::enums::*;
        #(#feature_code)*
    };

    let definition_code = quote! {
        use std::marker::PhantomData;
        use std::fmt;
        use std::os::raw::*;
        use crate::vk::{Handle, ptr_chain_iter};
        use crate::vk::aliases::*;
        use crate::vk::bitflags::*;
        use crate::vk::constants::*;
        use crate::vk::enums::*;
        use crate::vk::platform_types::*;
        use crate::vk::prelude::*;
        #(#definition_code)*
    };

    let enum_code = quote! {
        use std::fmt;
        #(#enum_code)*
        #core_debugs
    };

    let bitflags_code = quote! {
        use crate::vk::definitions::*;
        #(#bitflags_code)*
    };

    let constants_code = quote! {
        use crate::vk::definitions::*;
        #(#constants_code)*
    };

    let extension_code = quote! {
        use std::os::raw::*;
        use crate::vk::platform_types::*;
        use crate::vk::aliases::*;
        use crate::vk::bitflags::*;
        use crate::vk::definitions::*;
        use crate::vk::enums::*;
        #(#extension_code)*
    };

    let feature_extensions_code = quote! {
        use crate::vk::bitflags::*;
        use crate::vk::enums::*;
       #(#feature_extensions_code)*
    };

    let const_debugs = quote! {
        use std::fmt;
        use crate::vk::bitflags::*;
        use crate::vk::definitions::*;
        use crate::vk::enums::*;
        use crate::prelude::debug_flags;
        #const_debugs
    };

    let aliases = quote! {
        use crate::vk::bitflags::*;
        use crate::vk::definitions::*;
        use crate::vk::enums::*;
        #(#aliases)*
    };

    use std::io::Write;

    write!(&mut vk_features_file, "{}", feature_code).expect("Unable to write vk/features.rs");
    write!(&mut vk_definitions_file, "{}", definition_code)
        .expect("Unable to write vk/definitions.rs");
    write!(&mut vk_enums_file, "{}", enum_code).expect("Unable to write vk/enums.rs");
    write!(&mut vk_bitflags_file, "{}", bitflags_code).expect("Unable to write vk/bitflags.rs");
    write!(&mut vk_constants_file, "{}", constants_code).expect("Unable to write vk/constants.rs");
    write!(&mut vk_extensions_file, "{}", extension_code)
        .expect("Unable to write vk/extensions.rs");
    write!(
        &mut vk_feature_extensions_file,
        "{}",
        feature_extensions_code
    )
    .expect("Unable to write vk/feature_extensions.rs");
    write!(&mut vk_const_debugs_file, "{}", const_debugs)
        .expect("Unable to write vk/const_debugs.rs");
    write!(&mut vk_aliases_file, "{}", aliases).expect("Unable to write vk/aliases.rs");
}
