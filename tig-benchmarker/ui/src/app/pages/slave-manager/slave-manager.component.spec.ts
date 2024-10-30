import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SlaveManagerComponent } from './slave-manager.component';

describe('SlaveManagerComponent', () => {
  let component: SlaveManagerComponent;
  let fixture: ComponentFixture<SlaveManagerComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SlaveManagerComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SlaveManagerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
