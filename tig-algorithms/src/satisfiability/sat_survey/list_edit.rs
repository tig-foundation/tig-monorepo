// Straight-line forms of the unsat-list splice. The steps are the same
// swap-removes and appends as the counted loops, including their order.

#[inline(always)]
pub(crate) unsafe fn detach(list: &mut Vec<u32>, pos: &mut [u32], ids: *const u32, n: usize) {
    macro_rules! step {
        ($i:expr) => {{
            let c = *ids.add($i) as usize;
            let p = *pos.get_unchecked(c) as usize;
            let last_index = list.len() - 1;
            let last_clause = *list.get_unchecked(last_index) as usize;
            *list.get_unchecked_mut(p) = last_clause as u32;
            *pos.get_unchecked_mut(last_clause) = p as u32;
            list.set_len(last_index);
        }};
    }
    match n {
        0 => {}
        1 => { step!(0); }
        2 => { step!(0); step!(1); }
        3 => { step!(0); step!(1); step!(2); }
        4 => { step!(0); step!(1); step!(2); step!(3); }
        5 => { step!(0); step!(1); step!(2); step!(3); step!(4); }
        6 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); }
        7 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); }
        8 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); }
        9 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); }
        10 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); }
        11 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); }
        12 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); }
        13 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); }
        14 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); step!(13); }
        15 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); step!(13); step!(14); }
        16 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); step!(13); step!(14); step!(15); }
        _ => {
            for i in 0..n {
                let c = *ids.add(i) as usize;
                let p = *pos.get_unchecked(c) as usize;
                let last_index = list.len() - 1;
                let last_clause = *list.get_unchecked(last_index) as usize;
                *list.get_unchecked_mut(p) = last_clause as u32;
                *pos.get_unchecked_mut(last_clause) = p as u32;
                list.set_len(last_index);
            }
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn attach(list: &mut Vec<u32>, pos: &mut [u32], ids: *const u32, n: usize) {
    let len = list.len();
    macro_rules! step {
        ($i:expr) => {{
            let c = *ids.add($i);
            *pos.get_unchecked_mut(c as usize) = (len + $i) as u32;
            list.as_mut_ptr().add(len + $i).write(c);
        }};
    }
    match n {
        0 => {}
        1 => { step!(0); list.set_len(len + 1); }
        2 => { step!(0); step!(1); list.set_len(len + 2); }
        3 => { step!(0); step!(1); step!(2); list.set_len(len + 3); }
        4 => { step!(0); step!(1); step!(2); step!(3); list.set_len(len + 4); }
        5 => { step!(0); step!(1); step!(2); step!(3); step!(4); list.set_len(len + 5); }
        6 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); list.set_len(len + 6); }
        7 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); list.set_len(len + 7); }
        8 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); list.set_len(len + 8); }
        9 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); list.set_len(len + 9); }
        10 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); list.set_len(len + 10); }
        11 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); list.set_len(len + 11); }
        12 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); list.set_len(len + 12); }
        13 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); list.set_len(len + 13); }
        14 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); step!(13); list.set_len(len + 14); }
        15 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); step!(13); step!(14); list.set_len(len + 15); }
        16 => { step!(0); step!(1); step!(2); step!(3); step!(4); step!(5); step!(6); step!(7); step!(8); step!(9); step!(10); step!(11); step!(12); step!(13); step!(14); step!(15); list.set_len(len + 16); }
        _ => {
            for i in 0..n {
                let c = *ids.add(i);
                *pos.get_unchecked_mut(c as usize) = (len + i) as u32;
                list.as_mut_ptr().add(len + i).write(c);
            }
            list.set_len(len + n);
        }
    }
}
