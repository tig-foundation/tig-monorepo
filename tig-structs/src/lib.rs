pub mod config;
pub mod core;

#[macro_export]
macro_rules! serializable_struct_with_getters {
    ( @ $name:ident { } -> ($($fields:tt)*) ($($getters:tt)*) ) => (
        #[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
        pub struct $name {
            $($fields)*
        }
        impl $name {
            $($getters)*
        }
    );
    ( @ $name:ident { $(#[$attr:meta])* $param:ident : Option<$type:ty>, $($rest:tt)* } -> ($($fields:tt)*) ($($getters:tt)*) ) => (
        serializable_struct_with_getters!(@ $name { $($rest)* } -> (
            $($fields)*
            $(#[$attr])*
            #[serde(default)]
            pub $param : Option<$type>,
        ) (
            $($getters)*
            pub fn $param(&self) -> &$type {
                self.$param.as_ref().expect(
                    format!("Expected {}.{} to be Some, but it was None", stringify!($name), stringify!($param)).as_str()
                )
            }
        ));
    );

    ( @ $name:ident { $(#[$attr:meta])* $param:ident : $type:ty, $($rest:tt)* } -> ($($fields:tt)*) ($($getters:tt)*) ) => (
        serializable_struct_with_getters!(@ $name { $($rest)* } -> (
            $($fields)*
            $(#[$attr])*
            pub $param : $type,
        ) (
            $($getters)*
        ));
    );
    ( $name:ident { $( $rest:tt)* } ) => {
        serializable_struct_with_getters!(@ $name { $($rest)* } -> () ());
    };
}
