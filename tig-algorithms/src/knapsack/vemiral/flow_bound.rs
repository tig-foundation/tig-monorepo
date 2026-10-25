pub fn upper(profit:&[i64],weights:&[i32],capacity:i32,lambda:f64,arcs:impl Iterator<Item=(usize,usize,f64,i64)>)->i64 {
    let scale=65536i64;
    let penalty=(lambda*scale as f64).round().max(0.0) as i64;
    let mut balance=vec![0i64;profit.len()+2];
    let terminal:Vec<i64>=profit.iter().zip(weights).map(|(&p,&w)|scale*p-penalty*w as i64).collect();
    let constant:i64=terminal.iter().map(|&x|x.max(0)).sum();
    for (from,to,value,cap) in arcs {
        let capacity=if from==0 {terminal[to-2].max(0)}else if to==1 {(-terminal[from-2]).max(0)}else{scale*cap};
        let flow=(value*scale as f64).floor().max(0.0).min(capacity as f64) as i64;
        balance[from]+=flow;
        balance[to]-=flow;
    }
    let lower=(balance[0]-balance[2..].iter().map(|&x|(-x).max(0)).sum::<i64>()).max(0);
    (constant-lower+penalty*capacity as i64)/scale
}
